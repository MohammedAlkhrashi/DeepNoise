class Registry:
    def __init__(self) -> None:
        self._classes_dict = {}

    def register(self, name: str = None):
        def _register(cls):
            if name is not None:
                key = name
            else:
                key = cls.__name__
            key = key.lower()
            if key in self._classes_dict.keys():
                raise ValueError(f"A class with key: {name}, already registered")

            self._classes_dict[key] = cls
            return cls

        return _register

    def build(self, name, *args, **kwargs):
        key = name.lower()
        if key not in self._classes_dict.keys():
            raise ValueError(f"Type '{name}' is not registered.")

        cls = self._classes_dict[key]
        return cls(*args, **kwargs)

    def __contains__(self, key):
        return key.lower() in self._classes_dict
