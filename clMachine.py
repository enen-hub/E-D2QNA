import copy

class Machine:
    """Machine in factory - properties"""

    def __init__(self, aName, time, power_running=None, power_idle=None):
        self.name = aName
        self.currentTime = time
        self.assignedOpera = []
        self.runAning = 1

        # Add any other attributes that might exist
        self.status = getattr(self, 'status', 'normal')  # For fault tolerance
        self.fault_time = getattr(self, 'fault_time', None)
        self.repair_time = getattr(self, 'repair_time', None)


    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return self.name

    def exportToDict(self):
        """Serialize information about Machine into dictionary"""
        exData = {
            'machineName': self.name,
            'currentTime': self.currentTime,
            'assignedOper': self.assignedOpera,
            'status': getattr(self, 'status', 'normal'),
            'runAning': self.runAning
        }
        
        # Include fault-related attributes if present
        if hasattr(self, 'fault_time'):
            exData['fault_time'] = self.fault_time
        if hasattr(self, 'repair_time'):
            exData['repair_time'] = self.repair_time
            
        return exData