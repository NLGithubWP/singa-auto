import numpy as np
from cache import Cache
from common import REQUEST_QUEUE, INFERENCE_WORKER_SLEEP, BATCH_SIZE
import time

class InferenceWorker(object):

    def __init__(self, cache=Cache()):
        self._cache = cache
        self._load_model()

    def _load_model(self):
        #TODO: implement model
        self._model = model

    def start(self):
        while True:
            requests = self._cache.get_list_range(REQUEST_QUEUE, 0, BATCH_SIZE - 1)
            ids = []
            inputs = None

            for request in requests:
                if inputs is None:
                    inputs = request['input']
                else:
                    inputs = np.vstack([inputs, request['input']])
                ids.append(request['id'])
            
            if len(ids) > 0:
                predictions = self._model.predict(inputs)
                for (id, prediction) in zip(ids, predictions):
                    self._cache.append_list(id, prediction)

            time.sleep(INFERENCE_WORKER_SLEEP)

