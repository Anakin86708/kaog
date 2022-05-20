from threading import RLock


class ColunaYSingleton:
    _instance = None
    _nome_coluna_y = 'target'
    _lock = RLock()

    def __new__(cls, *args, **kwargs):
        if cls._instance:
            return cls._instance

        with cls._lock:
            cls._instance = super().__new__(cls, *args, **kwargs)
            return cls._instance

    @property
    def NOME_COLUNA_Y(self):
        return self._nome_coluna_y

    @NOME_COLUNA_Y.setter
    def NOME_COLUNA_Y(self, nome):
        self._nome_coluna_y = nome
