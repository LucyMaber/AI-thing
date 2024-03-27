# import aiomysql
# import asyncio
# import sqlite3
# import aiosqlite
# import pymongo
# import sys
# import os
# # import sqlite3
# # import motor.motor_asyncio
# # from milvus import Milvus
# import threading
# import motor.motor_tornado
# import multiprocessing
# from multiprocessing import Pool
# import random
# import re
# import aiosqlite
# URI = "mongodb://localhost:27017/"
# manager = multiprocessing.Manager()
# mogo_db = None
# mogo_motor_db = None
# veter_db = None


# async def make_veter_db():
#     return
#     global veter_db
#     veter_db = VeterDB()
#     return veter_db


# async def make_mogo_db():
#     global mogo_db
#     global mogo_motor_db
#     # mogo_db = MogoDB()
#     mogo_db = SQLiteDatabase()
#     await mogo_db.connect()
#     # mogo_motor_db = MogoDB_motor()
#     # mogo_motor_db = AioSQLiteDatabase()
#     return mogo_db


# class SQLiteDatabase:
#     def __init__(self, lock=None):
#         self.db = None
#         self.lock = lock

#     async def connect(self):
#         self.db = await aiosqlite.connect("./data/PPUK.db")
#         # try make tables
#         # try to make edge_entity
#         await self.db.execute('''CREATE TABLE IF NOT EXISTS edge_entity (
#             id integer PRIMARY KEY,
#             node_id text,
#             node_location text,
#             node_type text,
#             edge_entityid text,
#             edge_location text,
#             edge_type text,
#             edge_verb text,
#             version  text
#             )''')
#         await self.db.execute('''CREATE TABLE IF NOT EXISTS edge_time (
#             id integer PRIMARY KEY,
#             node_id text,
#             node_location text,
#             node_type text,
#             verb  text,
#             edge_verb text,
#             edge_location text,
#             edge_time  text,
#             edge_before  text,
#             edge_after  text,
#             edge_calendarmodel  text,
#             edge_precision  text,
#             edge_timezone  text,
#             edge_timecalendar  text,
#             edge_type  text,
#             version  text
#             )''')

#         # Create the edge_quantity table
#         await self.db.execute('''CREATE TABLE IF NOT EXISTS edge_quantity (
#         id integer PRIMARY KEY,
#         node_id text,
#         node_location text,
#         node_type text,
#         edge_type text,
#         edge_verb text,
#         amount text,
#         unit text,
#         upperBound text,
#         lowerBound text,
#         version text
#     )''')

#         # Create the edge_globecoordinate table
#         await self.db.execute('''CREATE TABLE IF NOT EXISTS edge_globecoordinate (
#         id integer PRIMARY KEY,
#         node_id text,
#         node_location text,
#         node_type text,
#         longitude text,
#         latitude text,
#         precision text,
#         version text
#          )''')

#         await self.db.execute('''CREATE TABLE IF NOT EXISTS edge_boolean (
#             id integer PRIMARY KEY,
#             node_id text,
#             node_location text,
#             node_type text,

#             edge_boolean boolean,
#             edge_location text,
#             edge_type text,
#             version text
#         )''')

#         await self.db.execute('''CREATE TABLE IF NOT EXISTS edge_text (
#             id integer PRIMARY KEY,
#             node_id text,
#             node_location text,
#             node_type text,
#             edge_verb text,

#             edge_text text,
#             edge_location text,
#             edge_type text,
#             version  text
#         )''')

#         # Create the edge_monolingualtext table (if not exists)
#         await self.db.execute('''CREATE TABLE IF NOT EXISTS edge_monolingualtext (
#             id integer PRIMARY KEY,
#             node_id text,
#             node_location text,
#             node_type text,
#             edge_text text,
#             edge_language text,
#             edge_verb text,
#             version text
#         )''')

#         # Create the edge_sitelink table (if not exists)
#         await self.db.execute('''CREATE TABLE IF NOT EXISTS edge_sitelink (
#             id integer PRIMARY KEY,
#             node_id text,
#             node_location text,
#             node_type text,
#             edge_text text,
#             edge_language text,
#             title text,
#             edge_verb text,
#             version text
#         )''')

#         # Create the edge_boolean table (if not exists)
#         await self.db.execute('''CREATE TABLE IF NOT EXISTS edge_boolean (
#             id integer PRIMARY KEY,
#             node_id text,
#             node_location text,
#             node_type text,
#             edge_boolean boolean,
#             edge_location text,
#             edge_type text,
#             version text
#         )''')
#         await self.db.execute('''CREATE TABLE IF NOT EXISTS mapping_text (
#             id integer PRIMARY KEY,
#             node_location text,

#             edge_type text,
#             edge_location text

#         )''')

#     async def graph_add_edges_to_node(self, node_id, where, type, edges, version=None):
#         # if self.lock is not None:
#         #     self.lock.acquire()
#         for edge in edges:
#             if "supertype" not in edge.keys():
#                 continue
#             try:
#                 if edge["supertype"] == "time":
#                     # print("start time")
#                     edge_where = edge["where"]
#                     edge_type = edge["type"]
#                     time = edge["object"]
#                     edge_verb = edge["verb"]
#                     # Check if the edge already exists
#                     try:
#                         cursor = await self.db.execute('''SELECT * FROM edge_time WHERE node_id= ? AND node_location= ? AND node_type= ? AND edge_time= ? AND edge_location= ? AND edge_type= ? AND edge_verb= ? AND version= ?''',
#                                                        (node_id, where, type, time, edge_where, edge_type, edge_verb, version))
#                         # if it does not exist, add it
#                         if not await cursor.fetchone():
#                             await self.db.execute('''
#                                             INSERT INTO edge_time(node_id, node_location, node_type, edge_time, edge_location, edge_type, edge_verb, version)
#                                             VALUES(?, ?, ?, ?, ?, ?, ?, ?)
#                                         ''', (node_id, where, type, time, edge_where, edge_type, edge_verb, version))
#                     except:
#                         pass
#                     # print("end time")
#                 elif edge["supertype"] == "location":
#                     # print("start time")
#                     # print("time:", edge)
#                     try:
#                         edge_type = edge["type"]
#                         edge_verb = edge["verb"]
#                         edge_location = edge["edge_location"]

#                         edge_time = edge["object"]["time"]
#                         edge_calendarmodel = edge["object"]["calendarmodel"]
#                         edge_timezone = edge["object"]["timezone"]
#                         edge_precision = edge["object"]["precision"]
#                         edge_after = edge["object"]["after"]
#                         edge_before = edge["object"]["before"]
#                         edge_timecalendar = edge["object"]["calendarmodel"]
#                         # Check if the edge already edbxists
#                         cursor = await self.db.execute('''SELECT * FROM edge_time WHERE node_id= ? AND node_location= ? AND node_type= ? AND edge_time= ? AND edge_location= ? AND edge_type= ? AND edge_verb= ? AND edge_before= ? AND edge_after= ? AND edge_precision= ? AND edge_timezone= ? AND edge_timecalendar= ? AND edge_calendarmodel= ? AND version = ?''',
#                                                        (node_id, where, type, edge_time, edge_where, edge_type, edge_verb, edge_before, edge_after, edge_precision, edge_timezone, edge_timecalendar, edge_calendarmodel, version))
#                         if not await cursor.fetchone():
#                             await self.db.execute('''
#                                 INSERT INTO edge_time(node_id, node_location, node_type, edge_verb, edge_time, edge_before, edge_after, edge_precision, edge_timezone, edge_timecalendar, edge_type, version)
#                                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#                             ''', (node_id, where, type, edge_verb, edge_time, edge_before, edge_after, edge_precision, edge_timezone, edge_timecalendar, edge_calendarmodel, edge_type, version))
#                     except:
#                         exc_type, exc_obj, exc_tb = sys.exc_info()
#                         print("time error:", exc_obj)
#                     # print("end entityid")
#                 elif edge["supertype"] == "quantity":
#                     # print("start quantity")
#                     try:
#                         edge_type = edge["type"]
#                         edge_verb = edge["verb"]
#                         amount = edge["object"]["amount"]
#                         edge_where = edge["unit"]
#                         if "upperBound" in edge:
#                             upperBound = edge["upperBound"]
#                         else:
#                             upperBound = None
#                         if "lowerBound" in edge:
#                             lowerBound = edge["lowerBound"]
#                         else:
#                             lowerBound = None

#                         cursor = await self.db.execute('''SELECT * FROM edge_quantity WHERE node_id = ? AND node_location = ? AND node_type = ? AND amount = ? AND unit = ? AND upperBound = ? AND lowerBound = ? AND version = ?''',
#                                                        (node_id, where, type, amount, edge_where, upperBound, lowerBound, version))
#                         if not await cursor.fetchone():
#                             await self.db.execute('''INSERT INTO edge_quantity(node_id, node_location, node_type, amount, unit, upperBound, lowerBound, version)
#                                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)
#                             ''', (node_id, where, type, amount, edge_where, upperBound, lowerBound, version))
#                     except:
#                         exc_type, exc_obj, exc_tb = sys.exc_info()
#                         print("quantity error:", exc_obj)
#                     # print("end quantity")

#                 elif edge["supertype"] == "string":
#                     # print("start string")
#                     edge_type = edge["type"]
#                     edge_verb = edge["verb"]
#                     edge_text = edge["value"]
#                     edge_where = edge["where"]
#                     try:
#                         cursor = await self.db.execute('''
#                             SELECT * FROM edge_text
#                             WHERE node_id = ? AND node_location = ? AND node_type = ? AND edge_text = ?
#                             AND edge_location = ? AND edge_type = ? AND version = ?
#                             ''',
#                                                        (node_id, where, type, edge_text,
#                                                         edge_where, edge_type, version)
#                                                        )
#                         if not await cursor.fetchone():
#                             await self.db.execute('''
#                                 INSERT INTO edge_text (node_id, node_location, node_type, edge_text, edge_location, edge_type, edge_verb, version)
#                                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)
#                                 ''', (node_id, where, type, edge_text, edge_where, edge_type, edge_verb, version)
#                             )
#                             await self.db.commit()  # Commit the changes to the database
#                     except Exception as e:
#                         print("string error:", e)
#                     # print("end string")
#                 elif edge["supertype"] == "globecoordinate":
#                     # print("start globecoordinate")
#                     try:
#                         # Handle globecoordinate data
#                         longitude = edge["object"]["longitude"]
#                         latitude = edge["object"]["latitude"]
#                         precision = edge["object"]["precision"]
#                         if longitude is not None:
#                             longitude = float(longitude)
#                         if latitude is not None:
#                             latitude = float(latitude)
#                         if precision is not None:
#                             precision = float(precision)

#                         cursor = await self.db.execute('''SELECT * FROM edge_globecoordinate WHERE node_id = ? AND node_location = ? AND node_type = ? AND longitude = ? AND latitude = ? AND precision = ? AND version = ?''',
#                                                        (node_id, where, type, longitude, latitude, precision, version))
#                         if not await cursor.fetchone():
#                             await self.db.execute('''INSERT INTO edge_globecoordinate(node_id, node_location, node_type, longitude, latitude, precision, version)
#                                 VALUES (?, ?, ?, ?, ?, ?, ?)
#                             ''', (node_id, where, type, longitude, latitude, precision, version))
#                     except:
#                         exc_type, exc_obj, exc_tb = sys.exc_info()
#                         print("globecoordinate error:", exc_obj)
#                     # print("end globecoordinate")
#                 elif edge["supertype"] == "monolingualtext":
#                     # print("start entityid")
#                     edge_verb = edge["verb"]
#                     edge_text = edge["value"]["text"]
#                     edge_language = edge["value"]["language"]
#                     # print("monolingualtext:", edge)
#                     try:
#                         # Handle monolingualtext data
#                         cursor = await self.db.execute('''SELECT * FROM edge_monolingualtext WHERE node_id = ? AND node_location = ? AND node_type = ? AND edge_text = ? AND edge_language = ? AND version = ?''',
#                                                        (node_id, where, type, edge_text, edge_language, version))
#                         if not await cursor.fetchone():
#                             await self.db.execute('''INSERT INTO edge_monolingualtext(node_id, node_location, node_type, edge_text, edge_language, edge_verb, version)
#                                 VALUES (?, ?, ?, ?, ?, ?, ?)
#                             ''', (node_id, where, type, edge_text, edge_language, edge_verb, version))
#                     except:
#                         exc_type, exc_obj, exc_tb = sys.exc_info()
#                         print("monolingualtext error:", exc_obj)
#                     # print("end monolingualtext")
#                 elif edge["supertype"] == "sitelink":
#                     # print("start sitelink")
#                     print("sitelink:", edge)
#                     edge_verb = edge["verb"]
#                     edge_text = edge["value"]["text"]
#                     title = edge["object"]
#                     language = edge["language"]
#                     try:
#                         # Handle sitelink data
#                         cursor = await self.db.execute('''SELECT * FROM edge_sitelink WHERE node_id = ? AND node_location = ? AND node_type = ? AND edge_text = ? AND edge_language = ? AND title = ? AND version = ?''',
#                                                        (node_id, where, type, edge_text, language, title, version))
#                         if not await cursor.fetchone():
#                             await self.db.execute('''INSERT INTO edge_sitelink(node_id, node_location, node_type, edge_text, edge_language, title, edge_verb, version)
#                                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)
#                             ''', (node_id, where, type, edge_text, language, title, edge_verb, version))
#                     except:
#                         exc_type, exc_obj, exc_tb = sys.exc_info()
#                         print("sitelink error:", exc_obj)
#                     # print("end sitelink")
#                 elif edge["supertype"] == "boolean":
#                     print("boolean:", edge)
#                     edge_verb = edge["verb"]
#                     edge_text = edge["value"]["text"]
#                     where = edge["where"]
#                     value = edge["object"]
#                     try:
#                         # Handle boolean data if necessary
#                         cursor = await self.db.execute('''SELECT * FROM edge_boolean WHERE node_id = ? AND node_location = ? AND node_type = ? AND edge_boolean = ? AND edge_location = ? AND edge_type = ? AND version = ?''',
#                                                        (node_id, where, type, value, where, edge_verb, version))
#                         if not await cursor.fetchone():
#                             await self.db.execute('''
#                                 INSERT INTO edge_boolean(node_id, node_location, node_type, edge_boolean, edge_location, edge_type, version)
#                                 VALUES (?, ?, ?, ?, ?, ?, ?)
#                             ''', (node_id, where, type, value, where, edge_verb, version))
#                     except:
#                         exc_type, exc_obj, exc_tb = sys.exc_info()
#                         print("boolean error:", exc_obj)
#                 else:
#                     print("edge:", edge)
#             except:
#                 exc_type, exc_obj, exc_tb = sys.exc_info()
#                 fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
#                 print("edge error_£", exc_type, fname, exc_tb.tb_lineno)
#                 print("error error_2:", e)

#     async def find_matching_entities(self, edge_location, edge_type, edge_text):
#         async with self.lock:  # Use the provided lock if necessary for concurrency control
#             # SQL query to find matching entities
#             query = """
#                 SELECT e.node_id, e.node_location, e.node_type
#                 FROM edge_entity AS e
#                 INNER JOIN mapping_entity AS me ON e.edge_location = me.edge_location
#                 INNER JOIN mapping_text AS mt ON e.edge_location = mt.edge_location
#                 WHERE
#                     (e.edge_location = ?  AND e.edge_type = ?   AND e.edge_text = ?) OR
#                     (me.edge_type = ? AND me.edge_type = ? AND e.edge_text = ?) OR
#                     (mt.edge_type = ? AND mt.edge_type = ? AND e.edge_text = ?) 
#             """
#             cursor = await self.db.execute(query, (edge_location, edge_type, edge_text, edge_type, edge_type))
#             result = await cursor.fetchall()
#             return result

#     async def find_matching_entities_partial(self, edge_location, edge_type, edge_text):
#         async with self.lock:
#             # Use '%' as a wildcard for partial matches
#             edge_text_partial = f"%{edge_text}%"
#             query = """
#                 SELECT e.node_id, e.node_location, e.node_type
#                 FROM edge_entity AS e
#                 INNER JOIN mapping_entity AS me ON e.edge_location = me.edge_location
#                 INNER JOIN mapping_text AS mt ON e.edge_location = mt.edge_location
#                 WHERE
#                     (e.edge_location = ? AND e.edge_type = ? AND e.edge_text LIKE ?) OR
#                     (me.edge_type = ? AND e.edge_location = ? AND e.edge_text LIKE ?) OR
#                     (mt.edge_type = ? AND e.edge_location = ? AND e.edge_text LIKE ?)
#             """
#             cursor = await self.db.execute(
#                 query, (edge_location, edge_type, edge_text_partial,
#                         edge_type, edge_text_partial, edge_type, edge_text_partial)
#             )
#             result = await cursor.fetchall()
#             return result

#     async def find_matching_entities_regex(self, edge_location, edge_type, edge_text):
#         async with self.lock:
#             query = """
#                 SELECT e.node_id, e.node_location, e.node_type
#                 FROM edge_entity AS e
#                 INNER JOIN mapping_entity AS me ON e.edge_location = me.edge_location
#                 INNER JOIN mapping_text AS mt ON e.edge_location = mt.edge_location
#                 WHERE
#                     (e.edge_location = ? AND e.edge_type = ? AND e.edge_text REGEXP ?) OR
#                     (me.edge_type = ? AND e.edge_location = ? AND e.edge_text REGEXP ?) OR
#                     (mt.edge_type = ? AND e.edge_location = ? AND e.edge_text REGEXP ?)
#             """
#             cursor = await self.db.execute(
#                 query, (edge_location, edge_type, edge_text,
#                         edge_type, edge_text, edge_type, edge_text)
#             )
#             result = await cursor.fetchall()
#             return result


# class VeterDB:
#     def __init__(self):
#         self.milvus = Milvus()
#         self.collection = "PPUK"

#     # Index the embedding in Milvus
#     def index_text_documents(self,  text_documents):
#         text_embedding_list = text_embedding.tolist()
#         status, ids = self.milvus.add_entity(
#             elf.collection, [text_embedding_list])
#         milvus_id = ids[0]
#         return milvus_id

#     def index_image_documents(self,  text_documents):
#         # Flatten and normalize the image features
#         image_embedding = image_features.squeeze().tolist()
#         # Index the embedding in Milvus
#         status, ids = self.milvus.add_entity(
#             self.collection.name, [image_embedding])
#         milvus_id = ids[0]
#         return milvus_id

#     def search_similar_text_documents(self, query_embedding, top_k=5):
#         pass

#     def search_similar_image_documents(self, query_embedding, top_k=5):
#         pass
