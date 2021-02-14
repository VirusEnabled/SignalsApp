import redis
import os
from redis import ConnectionError,TimeoutError
import json
from json import dumps as dump_json, loads as parse_json
from redis.client import string_keys_to_dict

def generate_redis_object():
	try:
		redis_db = redis.StrictRedis()
		return redis_db

	except ConnectionError:
		redis_db = redis.from_url(os.environ.get("REDIS_URL"))
		return redis_db

class RedisHandler(object):
	"""
	this handler will take care of centralizing the workload so that, we don't
	spread the same functions across the whole site.
	"""

	def __init__(self):
		self.client = generate_redis_object()

	def get_item(self, key: str) -> tuple:
		"""
		gets the value required for the
		existing key
		:param key:
		:return: tuple
		"""
		flag = False
		result = "There was an error with your request"
		try:
			value = self.client.get(key).decode()
			flag = True
			result = json.loads(value)

		except Exception as X:
			result = f"{result}: {X}"

		finally:
			return flag, result


	def load_value(self, key: str, value: dict) -> tuple:
		"""
		adds the value to a given key in
		redis caching
		:param key: str
		:param value: dict
		:return: tuple
		"""
		flag = False
		result = "There was an error with your request"
		try:
			self.client.set(key, json.dumps(value))
			flag = True
			result = 'Success'

		except Exception as X:
			result = f"{result}: {X}"

		finally:
			return flag, result
