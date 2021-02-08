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






if __name__ == '__main__':
	pass