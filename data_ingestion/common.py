from dataclasses import dataclass

@dataclass
class Data:
    id: str
    url: str
    title: str = ""
    date: str  = ""
    content: str = ""