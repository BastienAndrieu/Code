class Item:
    """
    Base class for items.
    Attributes:
    - name
    - use_in_battle: useable in battles
    - weight
    - nb_use_max: max number of uses (if reuseable, -1 means useable indefinitely)
    """
    def __init__(self,
                 name="",
                 use_in_battle=False,
                 weight=0,
                 nb_use_max=-1):
        self.name = name
        self.use_in_battle = use_in_battle
        self.weight = weight
        self.nb_use_max = nb_use_max
        return
    
###########################################

class Equipable(Item):
    """
    Base class for equipable items.
    Attributes:
    - delta_force
    - delta_speed
    - delta_precision
    - delta_defense
    """
    def __init__(self,
                 delta_force=0,
                 delta_speed=0,
                 delta_precision=0,
                 delta_defense=0):
        self.delta_force = delta_force
        self.delta_speed = delta_speed
        self.delta_precision = delta_precision
        self.delta_defense = delta_defense
        return

###########################################

class Weapon(Equipable):
    """
    Class for weapons.
    """
    def __init__(self, two_handed=False):
        self.two_handed = two_handed
        return

###########################################

class Wearable(Equipable):
    """
    Base class for wearables.
    Attributes:
    - gender :
    """
    def __init__(self, gender=None):
        self.gender = gender
        return
        
###########################################
        

    
class Inventary:
    def __init__(self, items=None, weightmax=0):
        self.items = items
        self.weightmax = weightmax
        return
