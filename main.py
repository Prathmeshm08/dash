import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import queue
import concurrent.futures
import time
import numpy as np
import subprocess
import logging
import shutil
import threading
from ultralytics import YOLO
from typing import Tuple, Dict, List
from datetime import datetime
from utils.Handlers.apiHandler import DASHCAMAPIHandler
from utils.Processors.VideoProcessing import VideoProcessor
from utils.Processors.detection_processor import DetectionProcessor
from utils.Processors.segmentation_processor import SegmentationProcessor
from utils.Handlers.detection_handler import DetectionGrouper
from utils.Handlers.segmentation_handler import SegmentationGrouper
from utils.Processors.ocr import FrameOCR
from db_manager import DBHandler
from utils.Processors.vectorDBProcessor import VectorDBSearch
from utils.Processors.classification_processor import ClassificationProcessor

# from viztracer import VizTracer

DEBUG = True

def setup_logging(survey_id):
    """Configure logging for each survey with timestamped log file."""
    os.makedirs("logs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"dashcam_log_survey_{survey_id}_{timestamp}.log"
    log_path = os.path.join("logs", log_filename)

    # Reset previous handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=log_path,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    # Add a console handler too (optional)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console_handler)

    # Add separator line for readability
    logging.info(f"\n----- Logging started for Survey ID: {survey_id} -----\n")



class Dashcam:
    def __init__(self, config, folder_path: str, surveyID,dayTime = "day"):
        """
        Initialize Dashcam system for processing multiple videos in a folder.
        """
        setup_logging(surveyID)
        try:   
            self.config = config
            self.detection_violation_classes = self.config["violation_classes"]["detection"]
            self.segmentation_violation_classes = self.config["violation_classes"]["segmentation"]
            self.segmentation_2_violation_classes = self.config["violation_classes"]["segmentation_2"]
            self.detection_model_path = self.config["models"]["detection_model_path"]
            self.segmentation_model_path = self.config["models"]["segmentation_model_path"]
            self.segmentation_2_model_path = self.config["models"]["segmentation_2_model_path"]
            self.classification_model_path = self.config["models"]["classification_model_path"]
            self.folder_path = folder_path
            self.surveyID = surveyID
            self.clip_saving_length = self.config.get("CLIP_SAVING_LENGTH", 10)
            self.local_botsort_path = self.config.get("BOTSORT_PATH", "botsort_local.yaml")
            self.segmentation_roi_path = self.config.get("SEGMENTATION_ROI_PATH", None)
            self.day_time = dayTime
            self.db_handler = DBHandler()

            enable_vdb = self.config.get("ENABLE_VECTOR_DB", False)

            if enable_vdb:
                path = self.config.get("VECTORDB_CONFIG_PATH", "")
                if path:
                    self.vector_db_processor = VectorDBSearch(path)
                else:
                    logging.warning("ENABLE_VECTOR_DB is True but VECTORDB_CONFIG_PATH is empty.")
                    self.vector_db_processor = None
            else:
                logging.info("VectorDBSearch disabled via config.")
                self.vector_db_processor = None

            

            # Validate folder path
            if not os.path.isdir(self.folder_path):
                raise NotADirectoryError(f"Provided folder path is invalid: {self.folder_path}")
            
            base_output_dir = self.config.get("data_saving_folder_path", None)
            if not base_output_dir:
                raise KeyError("Missing 'data_saving_folder_path' in config file.")

            # Create a timestamped parent folder inside that path
            current_time = datetime.now().strftime("%Y-%m-%d")
            self.parent_directory_name = current_time
            self.parent_output_dir = os.path.join(base_output_dir, self.parent_directory_name)

            # Create the survey folder inside it
            self.survey_output_dir = os.path.join(self.parent_output_dir, f"survey_{self.surveyID}")

            # Make sure both exist
            os.makedirs(self.survey_output_dir, exist_ok=True)

            # Log the creation
            if DEBUG:
                logging.info(f"Created survey directory: {self.survey_output_dir}")

            self.ocr_coordinates = self.config.get("ocr_coordinates", None)

            # Initialize queues for inter-thread communication
            self.API_posting_queue = queue.Queue()
            self.postprocessing_queue = queue.Queue()
            self.video_output_dir = os.path.join("results", "output_videos", f"survey_{self.surveyID}")
            # Initialize state flags
            self.processing_active = False
            self.stop_event = threading.Event()  

            # self.clip_processor = VideoProcessor(self.folder_path)

            

            self.video_processor = VideoProcessor(self.folder_path,survey_id=self.surveyID,db_handler=self.db_handler)
            self.detection_processor = DetectionProcessor(self.detection_model_path,
                                                        classes_list=self.detection_violation_classes,debug=DEBUG,
                                                        vector_db_processor=self.vector_db_processor)
            
            self.segmentation_processor = SegmentationProcessor(self.segmentation_model_path,
                                                                classes_list=self.segmentation_violation_classes,
                                                                debug=DEBUG,botsort_path=self.local_botsort_path, 
                                                                roi_path=self.segmentation_roi_path,
                                                                vector_db_processor=self.vector_db_processor,
                                                                separate_masks=False,
                                                                classification_processor=None)
            
            self.segmentation_2_processor = SegmentationProcessor(self.segmentation_2_model_path,
                                                                classes_list=self.segmentation_2_violation_classes,
                                                                debug=DEBUG,botsort_path=self.local_botsort_path,
                                                                vector_db_processor=self.vector_db_processor,
                                                                separate_masks=True,
                                                                classification_processor=ClassificationProcessor(self.classification_model_path))
            

            # self.ocr_processor = OCRProcessor(self.ocr_coordinates)
            self.ocr_processor = FrameOCR(self.ocr_coordinates,
                                          model_dir=self.config.get("ocr_model_dir", None))
            self.detection_results_handler = DetectionGrouper(debug=DEBUG)
            self.segmentation_results_handler = SegmentationGrouper(debug=DEBUG)
            self.api_handler = DASHCAMAPIHandler(self.config,self.db_handler)
            self.inference_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
            self.processing_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
            self._buffer_warmed_up = False
            
            # Rolling buffer setup (≈10 sec window)
            self.fps = 30
            self.frame_buffer = []  # half prefilled with None
            self.min_items_before_start = self.fps * self.clip_saving_length# e.g., 300 for 30fps * 10s

            self.cooldown_thresholds = self.config.get("cooldown_thresholds", {})
            self.cooldowns = {cls: 0 for cls in self.cooldown_thresholds.keys()} 

            logging.info(f"Dashcam initialized for folder: {self.folder_path}")

        except KeyError as e:
            logging.error(f"Missing configuration key: {e}")
            raise
        except Exception as e:
            logging.exception(f"Error initializing Dashcam: {e}")
            raise

    # centralized error propagation
    def handle_thread_exception(self, context: str, error: Exception):
        """Centralized fatal exception handler with safe shutdown."""
        logging.exception(f"Fatal error in {context}: {error}")
        
        self.stop_processing()
    
    # def generate_class_clip(
    #     self,
    #     class_name: str,
    #     source_type: str,
    #     output_path: str,
    #     fps: int = 30,
    #     scale_factor: float = 1.0,
    #     frame_count: int = 0,
    #     crf: int = 23
    # ):
    #     """
    #     Generate annotated video and compress directly to AVC1 (H.264)
    #     using GPU acceleration if available.
    #     """
    #     try:
    #         if not self.frame_buffer:
    #             logging.warning("Frame buffer is empty. Cannot generate class clip.")
    #             return False

    #         fps = fps or getattr(self, "fps", 30)
    #         max_frames = min(len(self.frame_buffer), fps * 10)
    #         frame_h, frame_w = self.frame_buffer[0]["frame"].shape[:2]
    #         new_w, new_h = int(frame_w * scale_factor), int(frame_h * scale_factor)
    #         os.makedirs(os.path.dirname(output_path), exist_ok=True)

    #         # Detect GPU encoder availability safely
    #         gpu_available = shutil.which("ffmpeg") and (
    #             subprocess.run(
    #                 ["ffmpeg", "-hide_banner", "-encoders"],
    #                 stdout=subprocess.PIPE,
    #                 stderr=subprocess.DEVNULL,
    #                 text=True
    #             ).stdout.find("h264_nvenc") != -1
    #         )

    #         encoder = "h264_nvenc" if gpu_available else "libx264"
    #         #preset = "p4" if encoder == "h264_nvenc" else "medium"
            
    #         if DEBUG:
    #             logging.info(
    #                 f"Using encoder: {encoder} | Resolution: {new_w}x{new_h} | FPS: {fps} | CRF: {crf}"
    #             )

    #         # # Build FFmpeg command (silent mode)
    #         # ffmpeg_cmd = [
    #         #     "ffmpeg", "-y",
    #         #     "-loglevel", "error",         # <— only show errors
    #         #     "-f", "rawvideo",
    #         #     "-vcodec", "rawvideo",
    #         #     "-pix_fmt", "bgr24",
    #         #     "-s", f"{new_w}x{new_h}",
    #         #     "-r", str(fps),
    #         #     "-i", "-",
    #         #     "-c:v", encoder,
    #         #     "-pix_fmt", "yuv420p",
    #         #     "-crf", str(crf),
    #         #     "-preset", preset,
    #         #     "-movflags", "+faststart",
    #         #     output_path
    #         # ]
    #         ffmpeg_cmd = [
    #             "ffmpeg", "-y",
    #             "-loglevel", "error",
    #             "-f", "rawvideo",
    #             "-vcodec", "rawvideo",
    #             "-pix_fmt", "bgr24",
    #             "-s", f"{new_w}x{new_h}",
    #             "-r", str(fps),
    #             "-i", "-",

    #             "-c:v", encoder,
    #             "-pix_fmt", "yuv420p",

    #             # ---------- HIGH QUALITY + FAST ----------
    #             "-rc", "vbr_hq",      # better quality rate control
    #             "-qmin", "1",         # allow high quality for complex areas
    #             "-qmax", "34",        # limit bad compression
    #             "-b:v", "1.2M",         # average bitrate
    #             "-maxrate", "2M",     # peak bitrate
    #             "-bufsize", "3M",
    #             "-preset", "p7" if encoder == "h264_nvenc" else "medium",
    #             # -----------------------------------------

    #             "-movflags", "+faststart",
    #             output_path
    #         ]


    #         # Launch FFmpeg with suppressed output
    #         process = subprocess.Popen(
    #             ffmpeg_cmd,
    #             stdin=subprocess.PIPE,
    #             stdout=subprocess.DEVNULL,
    #             stderr=subprocess.DEVNULL
    #         )

    #         annotated_count = 0

    #         for i, entry in enumerate(self.frame_buffer[:max_frames]):
    #             frame = entry["frame"].copy()
    #             source_data = entry.get(f"{source_type}_post") or {}

    #             if i==0 and DEBUG:
                    
    #                 logging.info(f"for frame count: {frame_count} starting frame is {entry['frame_count']}")
    #             elif i==max_frames-1 and DEBUG:
    #                 logging.info(f"for frame count: {frame_count} ending frame is {entry['frame_count']}")

    #             if isinstance(source_data, dict) and class_name in source_data:
    #                 class_info = source_data.get(class_name)
    #                 if class_info and "detections" in class_info:
    #                     for det in class_info["detections"]:
    #                         if not isinstance(det, dict) or "bbox" not in det:
    #                             continue
    #                         x1, y1, x2, y2 = map(int, det["bbox"])
    #                         color = (0, 255, 0) if source_type == "detection" else (0, 0, 255)
    #                         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    #                         cv2.putText(frame, class_name, (x1, max(y1 - 10, 20)),
    #                                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    #                     annotated_count += 1

    #             if scale_factor != 1.0:
    #                 frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    #             process.stdin.write(frame.tobytes())

    #         process.stdin.close()
    #         process.wait()

            
    #         logging.info(
    #             f"Compressed video saved: {output_path} | "
    #             f"Frames with '{class_name}': {annotated_count}/{max_frames} | Encoder: {encoder}"
    #         )
    #         return True

    #     except Exception as e:
    #         logging.error("Error in generate_class_clip", exc_info=True)
    #         return False

    


    def start_processing(self):
        """Start all threads for folder-based video processing."""
        logging.info(f"Starting Dashcam processing for folder: {self.folder_path}")
        self.processing_active = True

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                video_thread = executor.submit(self.start_video_processor_thread, executor)
                postprocessing_thread = executor.submit(self.start_post_processing_thread, executor)
                post_api_thread = executor.submit(self.send_violations_thread, executor)
                
                logging.info("All processing threads started successfully.")

                try:
                    while self.processing_active and not self.stop_event.is_set():
                        time.sleep(1)

                except KeyboardInterrupt:
                    logging.info("Processing interrupted by user (Ctrl+C). Stopping gracefully...")
                    self.stop_processing()
                except Exception as e:
                    self.handle_thread_exception("Main loop", e)
                finally:
                    self.stop_processing()
                    logging.info("Waiting for threads to complete...")

                    try:
                        video_thread.result(timeout=5)
                        postprocessing_thread.result(timeout=5)
                        post_api_thread.result(timeout=5)
                        logging.info("All threads have finished cleanly.")
                    except concurrent.futures.TimeoutError:
                        logging.warning("Some threads did not complete within timeout.")

        except Exception as e:
            self.handle_thread_exception("start_processing", e)
            self.stop_processing()

    def start_video_processor_thread(self, executor):
        """Thread: Read frames from videos and directly run model inference."""
        logging.info("Video processor thread started...")

        try:
            flag=self.video_processor.start()
            # print(flag)
            if not flag:
                logging.error("Failed to start VideoProcessor")
                self.postprocessing_queue.put(None)
                self.API_posting_queue.put(None)
                return

            # self.fps=self.video_processor.cap.get(cv2.CAP_PROP_FPS) or 30

            frame_count = 0
            while self.processing_active and not self.stop_event.is_set():
                try:
                    # print("reaching for frame data")
                    frame_data = self.video_processor.get_frame()
                    
                    if frame_data is None:
                        time.sleep(0.01)
                        # print("No frame data available")
                        continue


                    frame_count += 1
                    processed_frame_data = {
                        'frame': frame_data['frame'],
                        'frame_count': frame_data['frame_count'],
                        'timestamp': frame_data['timestamp'].strftime('%Y-%m-%d_%H-%M-%S'),
                        'video_name': frame_data.get('video_name', 'unknown')
                    }

                    # Directly run inference


                    future = executor.submit(self.run_model_inference, processed_frame_data)
                    future.result()
                

                except Exception as e:
                    self.handle_thread_exception("video processor thread", e)
                    break

            if self.video_processor:
                self.video_processor.stop()

            logging.info("Video processor thread completed.")

        except Exception as e:
            self.handle_thread_exception("video processor thread (outer)", e)

    def run_model_inference(self, frame_data):
        """Run detection, segmentation, and OCR on a single frame."""
        try:
            frame = frame_data["frame"]
            frame_count = frame_data["frame_count"]
            timestamp = frame_data["timestamp"]
            video_name = frame_data["video_name"]

            # print(frame_data)
           
            try:
                # futures = {
                #     "detection": self.inference_executor.submit(
                #         self.detection_processor.process_frame, frame
                #     ),
                #     "segmentation": self.inference_executor.submit(
                #         self.segmentation_processor.process_frame, frame
                #     )
                # }

                futures = {}
                # Always run detection
                futures["detection"] = self.inference_executor.submit(
                    self.detection_processor.process_frame, frame
                )

                # Run segmentation only during daytime
                if getattr(self, "day_time", "").lower() == "day":
                    if DEBUG:
                        logging.info("Running segmentation (day mode).")
                    futures["segmentation"] = self.inference_executor.submit(
                        self.segmentation_processor.process_frame, frame
                    )
                    
                    futures["segmentation_2"] = self.inference_executor.submit(
                        self.segmentation_2_processor.process_frame, frame
                    )
                    
                else:
                    if DEBUG:
                        logging.info(f"Skipping segmentation, day_time = {self.day_time}")

                # Optional OCR
                if getattr(self, "ocr_processor", None):
                    futures["ocr"] = self.inference_executor.submit(self.ocr_processor.process_frame, frame)

                results = {}
                for key, future in futures.items():
                    try:
                        results[key] = future.result()
                    except Exception as e:
                        logging.error(f"Error in {key} inference: {e}")
                        results[key] = None
                        raise  # propagate immediately

            except Exception as e:
                self.handle_thread_exception("run_model_inference local executor", e)
                return
    
            ocr_data = results.get("ocr") or {}

            processed_data = {
                "frame": frame,
                "frame_count": frame_count,
                "timestamp": timestamp,
                "video_name": video_name,
                "detection_result": results.get("detection"),
                "segmentation_result": results.get("segmentation")+results.get("segmentation_2"),
                "latitude": ocr_data.get("latitude"),
                "longitude": ocr_data.get("longitude"),
                "speed": ocr_data.get("speed"),
                "ocr_text": ocr_data.get("raw_text", "")
            }

            try:
                
                self.postprocessing_queue.put(processed_data, timeout=1)
            except queue.Full:
                logging.warning("Postprocessing queue full — dropping frame.")

        except Exception as e:
            self.handle_thread_exception("run_model_inference", e)


    def start_post_processing_thread(self, executor):
        """Thread: Handle detection + segmentation postprocessing and send to violation processor."""
        logging.info("Post-processing thread started.")
        try:
            while self.processing_active and not self.stop_event.is_set():  
                try:
                    
                    try:
                        data = self.postprocessing_queue.get(timeout=1)
                       
                    except queue.Empty:
                        continue
                    if data is None:
                        self.API_posting_queue.put(None)
                        break  # sentinel exit

                    frame = data["frame"]
                    frame_count = data["frame_count"]
                    timestamp = data["timestamp"]
                    video_name = data["video_name"]
                    if DEBUG:
                        print("Current frame:", frame_count, "Buffer length:", len(self.frame_buffer))
                    detection_result = data.get("detection_result")
                    segmentation_result = data.get("segmentation_result")
                    latitude = data.get("latitude")
                    longitude = data.get("longitude")
                    speed = data.get("speed")
                    ocr_text = data.get("ocr_text")

                    
                    try:
                        futures = {
                            "detection_post": self.processing_executor.submit(
                                self.detection_results_handler.handle_detection_result,
                                detection_result
                            ),
                            "segmentation_post": self.processing_executor.submit(
                                self.segmentation_results_handler.handle_segmentation_result,
                                segmentation_result
                            )
                        }

                        results = {}
                        for key, future in futures.items():
                            try:
                                results[key] = future.result(timeout=5)
                            except Exception as e:
                                logging.error(f"Error in {key}: {e}")
                                results[key] = None
                                raise  

                    except Exception as e:
                        self.handle_thread_exception("postprocessing local executor", e)
                        break

                    combined_data = {
                        "frame": frame,
                        "frame_count": frame_count,
                        "timestamp": timestamp,
                        "video_name": video_name,
                        "detection_post": results.get("detection_post"),
                        "segmentation_post": results.get("segmentation_post"),
                        "latitude": latitude,
                        "longitude": longitude,
                        "speed": speed,
                        "ocr_text": ocr_text,
                        "surveyID": self.surveyID
                    }

                    try:


                        buffer_entry = {
                            "frame": frame,
                            "frame_count": frame_count,
                            "detection_post": results.get("detection_post"),
                            "segmentation_post": results.get("segmentation_post")
                        }

                        # Add to shared frame buffer
                        self.frame_buffer.append(buffer_entry)
                        
                        if DEBUG:
                            print("Data added to frame buffer. Current length:", len(self.frame_buffer))

                        self.API_posting_queue.put(combined_data, timeout=1)
                    except queue.Full:
                        logging.warning("APIPosting queue full — dropping frame.")


                except Exception as e:
                    self.handle_thread_exception("postprocessing thread", e)
                    break

            logging.info("Post-processing thread completed.")

        except Exception as e:
            self.handle_thread_exception("postprocessing thread (outer)", e)


    def is_valid_gps(self, lat, lon):
        """Return True if the coordinates lie within India's bounding region."""
        if lat is None or lon is None:
            return False
        
        return (6.0 <= lat <= 37.1) and (68.0 <= lon <= 97.5)

    def send_violations_thread(self,executor):
        """
        Thread: Continuously consume violation data from the queue and send to API.
        Each queue item should be a 'combined_data' dict containing detection and segmentation results.
        """
        logging.info("Violation sending thread started.")
        

        if not self._buffer_warmed_up:
            sentinel_detected = False

            # Wait until queue fills or sentinel arrives
            while self.API_posting_queue.qsize() < self.min_items_before_start and not self.stop_event.is_set():
                # Peek at the first item without removing
                try:
                    first_item = self.API_posting_queue.queue[0]
                    if first_item is None:
                        logging.info("Sentinel detected during startup — skipping buffer warmup.")
                        sentinel_detected = True
                        break
                except IndexError:
                    pass  # queue empty, continue waiting

                logging.info(
                    f"Waiting for API queue to fill ({self.API_posting_queue.qsize()}/{self.min_items_before_start})..."
                )
                time.sleep(0.5)

            # If sentinel, skip the rest of warmup including the skip loop
            if not sentinel_detected and self.API_posting_queue.qsize() >= self.min_items_before_start:
                skip_count = self.min_items_before_start // 2
                logging.info(f"Initial fill complete — skipping {skip_count} oldest frames before starting processing.")

                for _ in range(skip_count):
                    try:
                        item = self.API_posting_queue.get_nowait()
                        if item is None:
                            logging.info("Sentinel detected during skip loop — breaking early.")
                            sentinel_detected = True
                            break
                    except queue.Empty:
                        break

            self._buffer_warmed_up = True

    
        try:
            
            while self.processing_active and not self.stop_event.is_set():
                try:
                    combined_data = self.API_posting_queue.get(timeout=1)
                except queue.Empty:
                    continue  # no data yet, loop again
                if DEBUG:
                    logging.info("Violation sending thread fetched data from queue.")

                if combined_data is None:
                    logging.info("Violation sending thread received stop signal.")
                    self.stop_processing()
                    break

                try:
                    frame = combined_data["frame"]
                    timestamp = combined_data["timestamp"]
                    frame_count = combined_data["frame_count"]
                    video_name = combined_data.get("video_name")
                    survey_id = combined_data.get("surveyID")
                    latitude = combined_data.get("latitude")
                    longitude = combined_data.get("longitude")
                    speed = combined_data.get("speed")
                    ocr_text = combined_data.get("ocr_text")

                    detection_results = combined_data.get("detection_post") or {}
                    segmentation_results = combined_data.get("segmentation_post") or {}

                    # ----- Prepare output directories -----
                    
                    annotated_img_dir = os.path.join(self.survey_output_dir, "images")
                    api_payload_dir = os.path.join(self.video_output_dir, "api_payload")
                    annotated_clip_dir = os.path.join(self.survey_output_dir, "videos")

                    os.makedirs(annotated_img_dir, exist_ok=True)
                    os.makedirs(api_payload_dir, exist_ok=True)
                    os.makedirs(annotated_clip_dir, exist_ok=True)

                    # ----- Combine detection & segmentation results -----
                    combined_results = {
                        "detection": detection_results,
                        "segmentation": segmentation_results
                    }

                    total_classes_sent = 0

                    # ----- Iterate over detection & segmentation -----
                    # ----- GPS VALIDATION -----
                    gps_valid = True

                    if gps_valid:
                        # ----- Iterate over detection & segmentation -----
                        for source_type, class_dict in combined_results.items():
                            if not isinstance(class_dict, dict):
                                continue

                            for class_name, instances in class_dict.items():
                                if not instances:
                                    continue

                                if self.cooldowns[class_name] == 0:

                                    total_classes_sent += 1
                                    logging.info(
                                        f"Processing {source_type} violations for class '{class_name}' "
                                        f"({len(instances)} instances)."
                                    )

                                    safe_timestamp = timestamp.replace(":", "-").replace(" ", "_")
                                    base_filename = f"{video_name}_{safe_timestamp}_frame{frame_count}_{class_name}"

                                    annotated_image_path = os.path.join(annotated_img_dir, f"{base_filename}.jpg")
                                    annotated_clip_path = os.path.join(annotated_clip_dir, f"{base_filename}.mp4")

                                    # generate video clip
                                    # self.generate_class_clip(
                                    #     class_name,
                                    #     source_type,
                                    #     annotated_clip_path,
                                    #     frame_count=frame_count
                                    # )

                                    violation_payload = {
                                        "timestamp": timestamp,
                                        "frame_count": frame_count,
                                        "survey_id": survey_id,
                                        "video_name": video_name,
                                        "latitude": latitude,
                                        "longitude": longitude,
                                        "speed": speed,
                                        "ocr_text": ocr_text,
                                        "source_type": source_type,
                                        "class_name": class_name,
                                        "instances": instances,
                                    }

                                    try:
                                        success = self.api_handler.process_and_send_violation(
                                            violations_data=violation_payload,
                                            frame=frame,
                                            annotated_frame_dir=annotated_img_dir,
                                            save_json_dir=api_payload_dir,
                                            annotated_image_path=annotated_image_path,
                                            annotated_clip_path=annotated_clip_path
                                        )

                                        if success:
                                            logging.info(f"Sent class '{class_name}' ({source_type}) successfully.")
                                        else:
                                            logging.warning(f"Failed to send class '{class_name}' ({source_type}).")

                                    except Exception as e:
                                        logging.error(f"Error sending '{class_name}' ({source_type}): {e}")

                                    self.cooldowns[class_name] = self.cooldown_thresholds.get(class_name, 60)
                    else:
                        logging.warning(
                            f"Skipping violation FOR-LOOPS for frame {frame_count} "
                            f"due to invalid GPS (lat={latitude}, lon={longitude})."
                        )


                    for cls in self.cooldowns:
                        if self.cooldowns[cls] > 0:
                            self.cooldowns[cls] -= 1

                    if DEBUG:
                        logging.info(f"trying to pop frame number from frame_buffer {frame_count} when the length is {len(self.frame_buffer)}")
                    
                    wait_start_time = time.time()
                    max_wait_time = 30  # seconds (adjust as needed)

                    while len(self.frame_buffer) < self.min_items_before_start:
                        if self.stop_event.is_set() or not self.processing_active:
                            logging.info("Stop event set or processing inactive — exiting waiting loop gracefully.")
                            return  # exit the thread cleanly

                        if time.time() - wait_start_time > max_wait_time:
                            logging.warning(
                                f"Timeout waiting for frame_buffer to reach {self.min_items_before_start}. "
                                f"Current length: {len(self.frame_buffer)}. Ending thread gracefully."
                            )
                            self.stop_processing(video_name)
                            return  # break out of the thread cleanly

                        logging.info(
                            f"Waiting for frame_buffer to have at least {self.min_items_before_start} items. "
                            f"Current length: {len(self.frame_buffer)}"
                        )
                        time.sleep(0.1)

                    try:
                        data = self.frame_buffer.pop(0)
                        if DEBUG:
                            logging.info(f"Popped oldest frame number {data['frame_count']} from frame_buffer after processing frame.")
                    except Exception as e:
                        logging.warning(f"Error popping from frame_buffer: {e}")

                except Exception as e:
                    logging.error(f"Error in send_violations_thread (processing item): {e}", exc_info=True)
                    continue

        except Exception as e:
            logging.error(f"Error in send_violations_thread main loop: {e}", exc_info=True)
        finally:
            logging.info("Violation sending thread stopped.")



    def stop_processing(self, video_name: str = None):
        """Stop all running threads and cleanup resources gracefully."""

        # Prevent double shutdown
        if self.stop_event.is_set():
            return

        logging.info("Stopping Dashcam processing...")

        # Signal everyone to stop
        self.processing_active = False
        self.stop_event.set()

        if video_name:
            logging.info("[DB] upating last video status to 'completed' in DB...")
            self.db_handler.update_status(video_name=video_name,survey_id=self.surveyID, new_status="completed",statusType="processed_status")

        # Unblock any thread waiting on queues
        for q in [self.API_posting_queue, self.postprocessing_queue]:
            try:
                q.put_nowait(None)  # sentinel to unblock
            except queue.Full:  
                pass

        # Stop the video processor and release capture
        try:
            if self.video_processor and hasattr(self.video_processor, "stop"):
                logging.info("Stopping video processor and releasing capture...")
                self.video_processor.stop()
        except Exception as e:
            logging.warning(f"Error stopping video processor: {e}")

        # Shutdown all thread pools safely
        try:
            if hasattr(self, "inference_executor"):
                logging.info("Shutting down inference executor...")
                self.inference_executor.shutdown(wait=True)

            if hasattr(self, "processing_executor"):
                logging.info("Shutting down processing executor...")
                self.processing_executor.shutdown(wait=True)

        except Exception as e:
            logging.warning(f"Error shutting down executors: {e}")

        logging.info("Shutdown sequence completed.")


# ---- Main entry point ----
if __name__ == "__main__":
    try:

        # Load configuration
        config_path = "config_day.json"
        if not os.path.exists(config_path):
            logging.error(f"Configuration file not found at path: {config_path}")
            raise FileNotFoundError(f"Missing configuration file: {config_path}")

        with open(config_path, "r") as f:
            try:
                config = json.load(f)
                logging.info("Configuration loaded successfully.")
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON: {e}")
                raise

        # Specify folder path
        folder_path = "video"
        if not os.path.exists(folder_path):
            logging.error(f"Video folder not found at path: {folder_path}")
            raise FileNotFoundError(f"Missing video folder: {folder_path}")

        # Create Dashcam object
        dashcam = Dashcam(config, folder_path, 40, "day")
        logging.info(f"Dashcam initialized successfully for folder: {folder_path}")


                # Start processing
        dashcam.start_processing()

    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")
        print(f"Error: {e}")
