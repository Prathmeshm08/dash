# # import sqlite3
# # import threading
# # import logging
# # import os

# # class DBHandler:
# #     """
# #     Handles video status database operations:
# #     - Create DB and table
# #     - Insert new videos
# #     - Update status after processing
# #     """

# #     def __init__(self, db_path="video_status.db"):
# #         self.db_path = db_path
# #         self.lock = threading.Lock()  # ensure thread-safety
# #         self._create_table()

# #     def _connect(self):
# #         """Create a new database connection."""
# #         return sqlite3.connect(self.db_path, check_same_thread=False)

# #     def _create_table(self):
# #         """Create table if it doesn't exist."""
# #         if "/" in self.db_path:
# #             os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

# #         conn = self._connect()
# #         cursor = conn.cursor()
# #         cursor.execute("""
# #             CREATE TABLE IF NOT EXISTS video_status (
# #                 id INTEGER PRIMARY KEY AUTOINCREMENT,
# #                 survey_id TEXT NOT NULL,
# #                 video_name TEXT NOT NULL UNIQUE,
# #                 status TEXT NOT NULL,
# #                 total_frame_count INTEGER NOT NULL,
# #                 recent_frame_count INTEGER NOT NULL
# #             )
# #         """)
# #         conn.commit()
# #         conn.close()
# #         logging.info("[DB] Initialized video_status table")

# #     def insert_video(self, survey_id: str, video_name: str, status: str = "pending"):
# #         """Insert a new video record (ignore if already exists)."""
# #         try:
# #             with self.lock:
# #                 conn = self._connect()
# #                 cursor = conn.cursor()
# #                 cursor.execute("""
# #                     INSERT OR IGNORE INTO video_status (survey_id, video_name, status)
# #                     VALUES (?, ?, ?)
# #                 """, (survey_id, video_name, status))
# #                 conn.commit()
# #                 conn.close()
# #             logging.info(f"[DB] Inserted or existing: {video_name} (survey_id={survey_id}, status={status})")
# #         except Exception as e:
# #             logging.error(f"[DB] Insert error for {video_name}: {e}")

# #     def update_status(self, video_name: str, new_status: str):
# #         """Update status for a video."""
# #         try:
# #             with self.lock:
# #                 conn = self._connect()
# #                 cursor = conn.cursor()
# #                 cursor.execute("""
# #                     UPDATE video_status
# #                     SET status = ?
# #                     WHERE video_name = ?
# #                 """, (new_status, video_name))
# #                 if cursor.rowcount == 0:
# #                     logging.warning(f"[DB] No record found for {video_name}, inserting instead.")
# #                     cursor.execute("""
# #                         INSERT INTO video_status (survey_id, video_name, status)
# #                         VALUES (?, ?, ?)
# #                     """, ("unknown", video_name, new_status))
# #                 conn.commit()
# #                 conn.close()
# #             logging.info(f"[DB] Updated {video_name} -> {new_status}")
# #         except Exception as e:
# #             logging.error(f"[DB] Update error for {video_name}: {e}")

# #     def update_total_frame_count(self, video_name: str, new_total_frame_count: int):
# #         """Update total_frame_count for a video."""
# #         try:
# #             with self.lock:
# #                 conn = self._connect()
# #                 cursor = conn.cursor()
# #                 cursor.execute("""
# #                     UPDATE video_status
# #                     SET total_frame_count = ?
# #                     WHERE video_name = ?
# #                 """, (new_total_frame_count, video_name))

# #                 # If the record doesn't exist, insert it
# #                 if cursor.rowcount == 0:
# #                     logging.warning(f"[DB] No record found for {video_name}, inserting instead.")
# #                     cursor.execute("""
# #                         INSERT INTO video_status (survey_id, video_name, status, total_frame_count, recent_frame_count)
# #                         VALUES (?, ?, ?, ?, ?)
# #                     """, ("unknown", video_name, "unknown", new_total_frame_count, 0))

# #                 conn.commit()
# #                 conn.close()

# #             logging.info(f"[DB] Updated total_frame_count for {video_name} -> {new_total_frame_count}")

# #         except Exception as e:
# #             logging.error(f"[DB] Update total_frame_count error for {video_name}: {e}")

# #     def get_total_frame_count(self, video_name: str) -> int:
# #         """Retrieve total_frame_count for a given video."""
# #         try:
# #             with self.lock:
# #                 conn = self._connect()
# #                 cursor = conn.cursor()
# #                 cursor.execute("""
# #                     SELECT total_frame_count
# #                     FROM video_status
# #                     WHERE video_name = ?
# #                 """, (video_name,))
# #                 result = cursor.fetchone()
# #                 conn.close()

# #             if result is not None:
# #                 total_frames = result[0]
# #                 logging.info(f"[DB] Retrieved total_frame_count for {video_name}: {total_frames}")
# #                 return total_frames
# #             else:
# #                 logging.warning(f"[DB] No record found for {video_name}")
# #                 return 0  # or None, depending on your preference

# #         except Exception as e:
# #             logging.error(f"[DB] Error retrieving total_frame_count for {video_name}: {e}")
# #             return 0


# #     def update_recent_frame_count(self, video_name: str, frame_count: int):
# #         """Update recent frame count for a video, insert if not exists."""
# #         try:
# #             with self.lock:
# #                 conn = self._connect()
# #                 cursor = conn.cursor()
# #                 cursor.execute("""
# #                     UPDATE video_status
# #                     SET recent_frame_count = ?
# #                     WHERE video_name = ?
# #                 """, (frame_count, video_name))

# #                 if cursor.rowcount == 0:
# #                     logging.warning(f"[DB] No record found for {video_name}, inserting instead.")
# #                     cursor.execute("""
# #                         INSERT INTO video_status (survey_id, video_name, status, recent_frame_count)
# #                         VALUES (?, ?, ?, ?)
# #                     """, ("unknown", video_name, "processing", frame_count))

# #                 conn.commit()
# #                 conn.close()

# #             logging.info(f"[DB] Updated recent_frame_count for {video_name} -> {frame_count}")

# #         except Exception as e:
# #             logging.error(f"[DB] Update recent_frame_count error for {video_name}: {e}")


# #     def check_video_exists(self, video_name: str) -> bool:
# #         """Check if a video already exists in the database."""
# #         with self.lock:
# #             conn = self._connect()
# #             cursor = conn.cursor()
# #             cursor.execute("SELECT COUNT(*) FROM video_status WHERE video_name = ?", (video_name,))
# #             exists = cursor.fetchone()[0] > 0
# #             conn.close()
# #         return exists

# #     def fetch_all(self):
# #         """Retrieve all video records (for debugging or reporting)."""
# #         with self.lock:
# #             conn = self._connect()
# #             cursor = conn.cursor()
# #             cursor.execute("SELECT survey_id, video_name, status FROM video_status")
# #             rows = cursor.fetchall()
# #             conn.close()
# #         return rows


# import sqlite3
# import threading
# import logging
# import os

# class DBHandler:
#     """
#     Thread-safe SQLite handler for video_status database.
#     """

#     def __init__(self, db_path="video_status.db"):
#         self.db_path = db_path
#         self.lock = threading.Lock()
#         self._create_table()

#     def _connect(self):
#         """Create a new database connection with a timeout."""
#         return sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)

#     def _create_table(self):
#         """Create table if it doesn't exist and enable WAL mode."""
#         if "/" in self.db_path:
#             os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

#         with self.lock, self._connect() as conn:
#             cursor = conn.cursor()
#             cursor.execute("PRAGMA journal_mode=WAL;")  # enable WAL
#             cursor.execute("""
#                 CREATE TABLE IF NOT EXISTS video_status (
#                     id INTEGER PRIMARY KEY AUTOINCREMENT,
#                     survey_id TEXT NOT NULL,
#                     video_name TEXT NOT NULL UNIQUE,
#                     status TEXT NOT NULL,
#                     total_frame_count INTEGER NOT NULL DEFAULT 0,
#                     recent_frame_count INTEGER NOT NULL DEFAULT 0
#                 )
#             """)
#             logging.info("[DB] Initialized video_status table")

#     def insert_video(self, survey_id: str, video_name: str, status: str = "pending"):
#         """Insert a new video record (ignore if already exists)."""
#         try:
#             with self.lock, self._connect() as conn:
#                 conn.execute("""
#                     INSERT OR IGNORE INTO video_status (survey_id, video_name, status)
#                     VALUES (?, ?, ?)
#                 """, (survey_id, video_name, status))
#             logging.info(f"[DB] Inserted or existing: {video_name} (survey_id={survey_id}, status={status})")
#         except Exception as e:
#             logging.error(f"[DB] Insert error for {video_name}: {e}")

#     def update_status(self, video_name: str, new_status: str):
#         try:
#             with self.lock, self._connect() as conn:
#                 cursor = conn.cursor()
#                 cursor.execute("""
#                     UPDATE video_status
#                     SET status = ?
#                     WHERE video_name = ?
#                 """, (new_status, video_name))
#                 if cursor.rowcount == 0:
#                     logging.warning(f"[DB] No record found for {video_name}, inserting instead.")
#                     cursor.execute("""
#                         INSERT INTO video_status (survey_id, video_name, status)
#                         VALUES (?, ?, ?)
#                     """, ("unknown", video_name, new_status))
#             logging.info(f"[DB] Updated {video_name} -> {new_status}")
#         except Exception as e:
#             logging.error(f"[DB] Update error for {video_name}: {e}")

#     def update_total_frame_count(self, video_name: str, new_total_frame_count: int):
#         try:
#             with self.lock, self._connect() as conn:
#                 cursor = conn.cursor()
#                 cursor.execute("""
#                     UPDATE video_status
#                     SET total_frame_count = ?
#                     WHERE video_name = ?
#                 """, (new_total_frame_count, video_name))
#                 if cursor.rowcount == 0:
#                     logging.warning(f"[DB] No record found for {video_name}, inserting instead.")
#                     cursor.execute("""
#                         INSERT INTO video_status (survey_id, video_name, status, total_frame_count, recent_frame_count)
#                         VALUES (?, ?, ?, ?, ?)
#                     """, ("unknown", video_name, "unknown", new_total_frame_count, 0))
#             logging.info(f"[DB] Updated total_frame_count for {video_name} -> {new_total_frame_count}")
#         except Exception as e:
#             logging.error(f"[DB] Update total_frame_count error for {video_name}: {e}")

#     def update_recent_frame_count(self, video_name: str, frame_count: int):
#         try:
#             with self.lock, self._connect() as conn:
#                 cursor = conn.cursor()
#                 cursor.execute("""
#                     UPDATE video_status
#                     SET recent_frame_count = ?
#                     WHERE video_name = ?
#                 """, (frame_count, video_name))
#                 if cursor.rowcount == 0:
#                     logging.warning(f"[DB] No record found for {video_name}, inserting instead.")
#                     cursor.execute("""
#                         INSERT INTO video_status (survey_id, video_name, status, recent_frame_count)
#                         VALUES (?, ?, ?, ?)
#                     """, ("unknown", video_name, "processing", frame_count))
#             logging.info(f"[DB] Updated recent_frame_count for {video_name} -> {frame_count}")
#         except Exception as e:
#             logging.error(f"[DB] Update recent_frame_count error for {video_name}: {e}")

#     def get_total_frame_count(self, video_name: str) -> int:
#         try:
#             with self.lock, self._connect() as conn:
#                 cursor = conn.cursor()
#                 cursor.execute("""
#                     SELECT total_frame_count
#                     FROM video_status
#                     WHERE video_name = ?
#                 """, (video_name,))
#                 result = cursor.fetchone()
#             return result[0] if result else 0
#         except Exception as e:
#             logging.error(f"[DB] Error retrieving total_frame_count for {video_name}: {e}")
#             return 0

#     def check_video_exists(self, video_name: str) -> bool:
#         try:
#             with self.lock, self._connect() as conn:
#                 cursor = conn.cursor()
#                 cursor.execute("SELECT COUNT(*) FROM video_status WHERE video_name = ?", (video_name,))
#                 return cursor.fetchone()[0] > 0
#         except Exception as e:
#             logging.error(f"[DB] Error checking video {video_name}: {e}")
#             return False

#     def fetch_all(self):
#         with self.lock, self._connect() as conn:
#             cursor = conn.cursor()
#             cursor.execute("SELECT survey_id, video_name, status FROM video_status")
#             return cursor.fetchall()

import sqlite3
import threading
import queue
import time
import os
import logging
from typing import Optional, Tuple, List, Any

class DBHandler:
    """
    DBHandler: single-writer SQLite handler with a queue.

    Usage:
        db = DBHandler("video_status.db")
        db.insert_video(survey_id, video_name, "pending")
        db.update_recent_frame_count(video_name, frame_no)
        last = db.get_total_frame_count(video_name)
        db.stop()  # graceful shutdown on exit
    """

    def __init__(
        self,
        db_path: str = "video_status.db",
        queue_maxsize: int = 10000,
        fallback_retries: int = 3,
        fallback_delay: float = 0.1,
    ):
        self.db_path = db_path
        self.queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._writer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False
        self.fallback_retries = fallback_retries
        self.fallback_delay = fallback_delay

        # Ensure folder exists
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        # Remove stale temporary files left from previous crashes (safe)
        # for ext in ("-wal", "-shm", "-journal"):
        #     p = f"{self.db_path}{ext}"
        #     if os.path.exists(p):
        #         try:
        #             os.remove(p)
        #             logging.info(f"[DB] Removed stale file: {p}")
        #         except Exception:
        #             logging.warning(f"[DB] Could not remove stale file: {p}")

        # Initialize DB (creates table and configures WAL)
        self._init_db()

        # Start writer thread
        self._start_writer()

    # -------------------------
    # Internal helpers
    # -------------------------
    def _connect_for_writer(self) -> sqlite3.Connection:
        # Single persistent connection used by the writer thread
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        # Prefer WAL for concurrency (readers vs single writer). Keep busy timeout.
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")  # 5s
        except Exception as e:
            logging.warning(f"[DB] Warning setting pragmas: {e}")
        return conn

    def _connect(self) -> sqlite3.Connection:
        # For direct-read/direct-fallback writes — short timeout
        return sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)

    def _init_db(self):
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS video_status (
                    survey_id TEXT NOT NULL,
                    video_name TEXT NOT NULL,
                    read_status TEXT NOT NULL,
                    processed_status TEXT NOT NULL,
                    final_status TEXT NOT NULL,
                    total_frame_count INTEGER NOT NULL DEFAULT 0,
                    recent_frame_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (survey_id, video_name)
                );

            """)
            conn.commit()
            conn.close()
            logging.info("[DB] Initialized database and table.")
        except Exception as e:
            logging.exception(f"[DB] Failed to initialize DB: {e}")
            raise

    def _start_writer(self):
        if self._started:
            return
        self._stop_event.clear()
        self._writer_thread = threading.Thread(target=self._writer_loop, name="DBWriter", daemon=True)
        self._writer_thread.start()
        self._started = True
        logging.info("[DB] DB writer thread started.")

    def _writer_loop(self):
        """
        Writer thread: consumes tasks from self.queue and writes to SQLite.
        Task format: (sql, params_tuple)
        A sentinel value of None -> shutdown.
        """
        conn = None
        try:
            conn = self._connect_for_writer()
            cur = conn.cursor()
            while not self._stop_event.is_set():
                try:
                    task = self.queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                # Shutdown sentinel
                if task is None:
                    self.queue.task_done()
                    break

                sql, params = task
                try:
                    cur.execute(sql, params)
                    logging.info(f"[DB] Writer executed SQL: {sql} | params: {params}")
                    conn.commit()
                except sqlite3.OperationalError as e:
                    # writer thread should rarely hit this, but handle gracefully
                    logging.error(f"[DB] Writer OperationalError executing SQL: {e} | SQL: {sql} | params: {params}")
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                except Exception as e:
                    logging.exception(f"[DB] Writer unexpected error: {e}")
                finally:
                    # mark task done even in error so queue.join() doesn't block forever
                    try:
                        self.queue.task_done()
                    except Exception:
                        pass

            logging.info("[DB] Writer loop exiting.")
        except Exception as e:
            logging.exception(f"[DB] Fatal error in writer thread init: {e}")
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
            logging.info("[DB] Writer connection closed.")

    # -------------------------
    # Public enqueue helpers
    # -------------------------
    def _enqueue(self, sql: str, params: Tuple[Any, ...]):
        """
        Enqueue a write task. If queue is full, attempt fallback direct-write for a few retries.
        """
        if self._stop_event.is_set():
            # writer stopping — perform direct write as fallback
            logging.warning("[DB] Writer stopping — performing fallback direct write.")
            return self._direct_write_with_retry(sql, params)

        try:
            # block briefly if queue is almost full, but don't block forever
            self.queue.put((sql, params), block=True, timeout=0.5)
            return True
        except queue.Full:
            # fallback path: do direct writes with retries so caller is not blocked forever
            logging.warning("[DB] DB queue full — performing fallback direct write.")
            return self._direct_write_with_retry(sql, params)

    def _direct_write_with_retry(self, sql: str, params: Tuple[Any, ...]) -> bool:
        """
        Fallback direct write if queue is full or writer is stopped.
        Retries a few times with small delays.
        """
        for attempt in range(self.fallback_retries):
            try:
                conn = self._connect()
                cur = conn.cursor()
                cur.execute(sql, params)
                conn.commit()
                conn.close()
                logging.info("[DB] Fallback direct write succeeded.")
                return True
            except sqlite3.OperationalError as e:
                logging.warning(f"[DB] Direct write OperationalError (attempt {attempt+1}): {e}")
                time.sleep(self.fallback_delay)
            except Exception as e:
                logging.exception(f"[DB] Direct write error (attempt {attempt+1}): {e}")
                time.sleep(self.fallback_delay)
        logging.error("[DB] Direct write failed after retries.")
        return False

    # -------------------------
    # Public API (enqueue operations)
    # -------------------------
    def insert_video(self, survey_id: str, video_name: str, status: str = "pending") -> bool:
        sql = """
            INSERT OR IGNORE INTO video_status
            (survey_id, video_name, read_status, processed_status, final_status, total_frame_count, recent_frame_count)
            VALUES (?, ?, ?, ?, ?, 0, 0)
        """
        return self._enqueue(sql, (survey_id, video_name, status, status, status))


    # def update_status(self, video_name: str, survey_id: str, statusType: str, new_status: str) -> bool:
    #     """Generic status updater with 'all' support."""
        
    #     valid_columns = {"read_status", "processed_status", "final_status", "all"}
    #     if statusType not in valid_columns:
    #         raise ValueError(f"Invalid status type: {statusType}")

    #     if statusType == "all":
    #         sql = ("UPDATE video_status SET read_status = ?, processed_status = ?, final_status = ? "
    #             "WHERE survey_id = ? AND video_name = ?")
    #         return self._enqueue(sql, (new_status, new_status, new_status, survey_id, video_name))

    #     sql = f"UPDATE video_status SET {statusType} = ? WHERE survey_id = ? AND video_name = ?"
    #     return self._enqueue(sql, (new_status, survey_id, video_name))

    def update_status(self, video_name: str, survey_id: str, statusType: str, new_status: str) -> bool:
        """Update a specific status field OR all fields.
        Also auto-update final_status when read + processed are completed.
        """

        valid_columns = {"read_status", "processed_status", "final_status", "all"}
        if statusType not in valid_columns:
            raise ValueError(f"Invalid status type: {statusType}")

        # -----------------------------------------
        # 1️⃣ Handle statusType = "all"
        # -----------------------------------------
        if statusType == "all":
            sql = """
                UPDATE video_status
                SET read_status = ?, processed_status = ?, final_status = ?
                WHERE survey_id = ? AND video_name = ?
            """
            return self._enqueue(sql, (new_status, new_status, new_status, survey_id, video_name))

        # -----------------------------------------
        # 2️⃣ Update only the requested column
        # -----------------------------------------
        sql = f"UPDATE video_status SET {statusType} = ? WHERE survey_id = ? AND video_name = ?"
        updated = self._enqueue(sql, (new_status, survey_id, video_name))

        # If updating final_status directly → no trigger logic
        if statusType == "final_status":
            return updated

        # -----------------------------------------
        # 3️⃣ Trigger logic: if read + processed completed → final_status = completed
        # -----------------------------------------
        try:
            conn = self._connect()
            cur = conn.cursor()

            cur.execute("""
                SELECT read_status, processed_status
                FROM video_status
                WHERE survey_id = ? AND video_name = ?
            """, (survey_id, video_name))

            row = cur.fetchone()
            conn.close()

            if not row:
                return updated

            read_s, proc_s = row

            if read_s == "completed" and proc_s == "completed":
                logging.info(f"[DB] Auto-updating final_status to 'completed' for {video_name}")
                sql_final = """
                    UPDATE video_status
                    SET final_status = 'completed'
                    WHERE survey_id = ? AND video_name = ?
                """
                self._enqueue(sql_final, (survey_id, video_name))

        except Exception as e:
            logging.exception(f"[DB] Error in auto-final update for {video_name}: {e}")

        return updated



    def update_total_frame_count(self, video_name: str, total_frames: int) -> bool:
        sql = "UPDATE video_status SET total_frame_count = ? WHERE video_name = ?"
        return self._enqueue(sql, (total_frames, video_name))

    def update_recent_frame_count(self, video_name: str, frame_count: int) -> bool:
        sql = "UPDATE video_status SET recent_frame_count = ? WHERE video_name = ?"
        return self._enqueue(sql, (frame_count, video_name))

    # -------------------------
    # Read operations — direct (not enqueued)
    # -------------------------
    def get_total_frame_count(self, video_name: str) -> int:
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("SELECT total_frame_count FROM video_status WHERE video_name = ?", (video_name,))
            row = cur.fetchone()
            conn.close()
            return int(row[0]) if row and row[0] is not None else 0
        except Exception as e:
            logging.exception(f"[DB] Error getting total_frame_count: {e}")
            return 0

    def get_recent_frame_count(self, video_name: str) -> int:
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("SELECT recent_frame_count FROM video_status WHERE video_name = ?", (video_name,))
            row = cur.fetchone()
            conn.close()
            return int(row[0]) if row and row[0] is not None else 0
        except Exception as e:
            logging.exception(f"[DB] Error getting recent_frame_count: {e}")
            return 0

    def check_video_exists(self, video_name: str) -> bool:
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM video_status WHERE video_name = ? LIMIT 1", (video_name,))
            exists = cur.fetchone() is not None
            conn.close()
            return exists
        except Exception as e:
            logging.exception(f"[DB] Error checking exists: {e}")
            return False

    def fetch_all(self) -> List[Tuple]:
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("SELECT survey_id, video_name, status, total_frame_count, recent_frame_count FROM video_status")
            rows = cur.fetchall()
            conn.close()
            return rows
        except Exception as e:
            logging.exception(f"[DB] Error fetching all rows: {e}")
            return []
        
    def get_status(self, video_name: str, survey_id: str, statusType: str) -> str:
        """Generic getter for any status column."""
        
        valid_columns = {"read_status", "processed_status", "final_status"}
        if statusType not in valid_columns:
            raise ValueError(f"Invalid status type: {statusType}")

        try:
            conn = self._connect()
            cur = conn.cursor()
            query = f"SELECT {statusType} FROM video_status WHERE survey_id = ? AND video_name = ?"
            cur.execute(query, (survey_id, video_name))
            
            row = cur.fetchone()
            conn.close()

            return str(row[0]) if row and row[0] is not None else "pending"

        except Exception as e:
            logging.exception(f"[DB] Error getting {statusType} for {video_name}: {e}")
            return "pending"



    # -------------------------
    # Shutdown / flushing
    # -------------------------
    def flush(self, timeout: Optional[float] = None) -> bool:
        """
        Block until the queue is emptied (or timeout seconds elapse).
        Returns True if queue emptied, False if timed out.
        """
        start = time.time()
        while not self.queue.empty():
            if timeout is not None and (time.time() - start) > timeout:
                return False
            time.sleep(0.05)
        return True

    def stop(self, wait_timeout: float = 5.0):
        """
        Graceful shutdown: tell writer thread to stop after finishing queued tasks.
        Blocks up to wait_timeout seconds for writer thread to join.
        """
        if not self._started:
            return

        logging.info("[DB] Stopping DBHandler: waiting for queue to drain.")
        # Wait briefly for queue to drain
        self.flush(timeout=5.0)

        # send sentinel to writer thread to exit
        try:
            self.queue.put_nowait(None)
        except Exception:
            # If queue full/unavailable, set stop flag to force writer to end.
            pass

        self._stop_event.set()
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=wait_timeout)
        self._started = False
        logging.info("[DB] DBHandler stopped.")

    # context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.stop()
        except Exception:
            pass
