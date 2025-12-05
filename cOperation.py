
from typing import Dict, Optional
from core.clItinerary import Itinerary
class Operation():  
    def __init__(self, parent_job: Itinerary, operation_id: int, 
                 candidate_machines: Dict[str, int]):
        # --- 核心身份 ---
        self.parent_job = parent_job          
        self.idItinerary = parent_job.idItinerary  
        self.itinerary = parent_job.itinerary
        self.idOperation = int(operation_id)
        self.machine = candidate_machines
        self.due_date = parent_job.due_date
        
        # --- 调度状态 ---
        self.startTime = 0
        self.duration = 0
        self.endTime = 0
        self.completed = False
        self.assignedMachine: Optional[str] = None
        
        # --- 辅助属性 ---
        self.colorOfItinerary = None
        self.priority = 0
        self.completedRatio = 0.0
        self.remainingOps = 0
    
    @property
    def job_id(self) -> str:
        
        return self.parent_job.id
    
    def exportToDict(self):
        return {
            "itinerary": self.itinerary,
            "idItinerary": self.idItinerary,
            "idOperation": self.idOperation,
            "machine": self.machine,
            "duration": self.duration,
            "startTime": self.startTime,
            "endTime": self.endTime,
            "completed": self.completed,
            "assignedMachine": self.assignedMachine,
            "due_date": self.due_date,
            "colorOfItinerary": self.colorOfItinerary,
            "priority": self.priority,
            "completedRatio": self.completedRatio,
            "remainingOps": self.remainingOps,
        }
    
    def __eq__(self, other):
        return (self.itinerary == other.itinerary and 
                self.idOperation == other.idOperation)
    
    def __hash__(self):
        return hash((self.itinerary, self.idOperation))
    
    def __str__(self):
        return f"Op_{self.idItinerary}_{self.idOperation}"
    
    def getTupleStartAndDuration(self):
        return (self.startTime, self.duration)
    
    def getEndTime(self):
        self.endTime = self.startTime + self.duration
        return self.endTime


