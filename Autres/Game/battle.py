import character

class Loot:
    """
    Class for loots (things the party gets when a battle is won)
    """
    def __init__(self, xp=0, gold=0, items=None):
        self.xp = xp
        self.gold = gold
        self.items = items
        return

###########################################

def assign_loot(party, loot, verbose=True):
    """
    Distribute a loot over all the members of a party.
    """
    nheroes = len(party.members)
    assert (nheroes > 0), "The party has no members!"
    # distribute xp gain evenly over all the party members
    xp_per_hero = int(loot.xp/float(nheroes))
    for hero in party.members:
        stat = hero.gain_xp(xp_per_hero, verbose)
    # gain gold
    stat = party.gain_gold(loot.gold, verbose)
    return True
            
    
    
