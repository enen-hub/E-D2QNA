from typing import List, Dict, Optional

class Itinerary():
    def __init__(self, job_id: str, due_date: int):
        self.id = str(job_id)
        self.name = f"Job{self.id}"
        self.due_date = due_date
        self.operations: List['Operation'] = []
    
    @property
    def is_completed(self) -> bool:
        if not self.operations:
            return True
        return all(op.completed for op in self.operations)
    
    def add_operation(self, operation: 'Operation'):
        self.operations.append(operation)
    
    def __str__(self):
        return f"Job(id={self.id}, due={self.due_date}, ops={len(self.operations)})"
    
    def exportToDict(self):
        return {
            'itineraryName': self.name,
            'due_date': self.due_date,
            'operations': [op.exportToDict() for op in self.operations]
        }
    
    @property
    def itinerary(self) -> str:
        return self.name
    @property
    def idItinerary(self) -> int:
        try:
            return int(self.id)
        except Exception:
            return 0