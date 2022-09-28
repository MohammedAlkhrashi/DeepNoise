class Registry:
    def __init__(self) -> None:
        self._classes_dict = {}

    def register(self, name=None):
        def _register(cls):
            if name is not None:
                key = name
            else:
                key = cls.__name__

            if key in self._classes_dict.keys():
                raise ValueError(f"A class with key: {name}, already registered")

            self._classes_dict[key] = cls
            return cls

        return _register

    def build(self, name, *args, **kwargs):
        if name not in self._classes_dict.keys():
            raise ValueError(f"Class key, {name}, does not exist.")

        cls = self._classes_dict[name]
        return cls(*args, **kwargs)
