class ColunaYSingleton:

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ColunaYSingleton, cls).__new__(cls)
            cls._nome_coluna_y = 'target'
        return cls.instance

    @property
    def NOME_COLUNA_Y(self):
        return self._nome_coluna_y

    @NOME_COLUNA_Y.setter
    def NOME_COLUNA_Y(self, nome):
        self._nome_coluna_y = nome
