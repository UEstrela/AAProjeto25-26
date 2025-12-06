class Action:
    """
    Representa uma ação genérica que um agente pode realizar.
    """
    def __init__(self, name: str, value: any = None):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"Action({self.name}, {self.value})"

class Observation:
    """
    Representa a perceção que o agente tem do mundo num dado momento.
    Pode conter vetores, matrizes ou objetos simples.
    """
    def __init__(self, data: any):
        self.data = data
        self.timestamp = 0  # Pode ser útil para debugging

    def __repr__(self):
        return f"Observation({self.data})"