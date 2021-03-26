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
			value = self.client.get(key)
			if value:
				flag = True
				result = json.loads(value.decode())
			else:
				result+=f": The key: {key} doesn't exist, try a different one."

		except Exception as X:
			result = f"{result}: {X}"

		finally:
			return flag, result


	def load_value(self, key: str, value: object) -> tuple:
		"""
		adds the value to a given key in
		redis caching

		this takes care to update the value if the value already exists
		since we're going to deal with multiple objects and live system
		therefore it needs to handle overriding.
		:param key: str
		:param value: dict
		:return: tuple
		"""
		flag = False
		result = "There was an error with your request"
		try:

			flag, existing_data = self.get_item(key)
			if flag and isinstance(existing_data,list) and isinstance(value,list):
				new = existing_data + value
				self.client.set(key, new)

			elif flag and isinstance(existing_data, dict) and isinstance(value, dict):
				for key in value.keys():
					existing_data[key] = value[key]
				self.client.set(key, existing_data)

			elif not flag and "doesn't exist" in existing_data:
				self.client.set(key, json.dumps(value))

			else:
				raise AttributeError(existing_data)

			flag = True
			result = 'Success'
		except Exception as X:
			result = f"{result}: {X}"

		finally:
			return flag, result


	def save_historical_data(self, user: object, symbol: str,
							  data: list) -> tuple:
		"""
		saves the historical data as a whole rather than
		one by one
		:param user: Userobject, normally the logged user
		:param symbol: str: the symbol's data
		:param data: list of dict containing the historical data
		from the symbol provided
		:return: tutple
		"""
		flag = False
		result = "There was an error with your request"
		try:
			key = f"historical_for_{symbol}_by_{user.username}"
			flag, result = self.load_value(key, data)

		except Exception as X:
			result+=f": {X}"

		finally:
			return flag, result



	def retrieve_historical_data(self,  user: object, symbol: str) -> tuple:
		"""
		retrieves the historical data belonging to the existing user
		:param user: object
		:param symbol: str
		:return: tuple
		"""
		flag = False
		result = "There was an error with your request"
		try:
			key = f"historical_for_{symbol}_by_{user.username}"
			flag, result = self.get_item(key)

		except Exception as X:
			result+=f": {X}"

		finally:
			return flag, result


	def save_graph_refresh_time(self, user: object, symbol: str, refresh_time: str) -> tuple:
		"""
		saves the historical data as a whole rather than
		one by one
		:param user: Userobject, normally the logged user
		:param symbol: str: the symbol's data
		:param refresh_time: str: date and timestamp of the operation
		from the symbol provided
		:return: tutple
		"""
		flag = False
		result = "There was an error with your request"
		try:
			key =f"last_graph_refresh_time_{user.username}_for_{symbol}"
			self.client.set(key, json.dumps(refresh_time))
			flag = True
			result = f"Success{ json.dumps(refresh_time)}"
		except Exception as X:
			result+=f": {X}"

		finally:
			return flag, result



	def get_graph_refresh_time(self, user: object, symbol: str) -> tuple:
		"""
		retrieves the last time the graph was updated data belonging to the existing user
		:param user: Userobject, normally the logged user
		:param symbol: str: the symbol's data
		from the symbol provided
		:return: tutple
		"""
		flag = False
		result = "There was an error with your request"
		try:
			key =f"last_graph_refresh_time_{user.username}_for_{symbol}"
			flag, result = self.get_item(key)

		except Exception as X:
			result+=f": {X}"

		finally:
			return flag, result
