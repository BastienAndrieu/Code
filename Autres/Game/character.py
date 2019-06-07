"""
TO DO
support battle with mutliple enemies (enemy party?)
"""




class Character:
    """
    Base class for characters.
    Attributes:
    - name
    - gender ('M', 'F', None)
    """
    def __init__(self, name="", gender=None):
        self.name = name
        self.gender = gender
        return
    ###
    def is_male(self):
        return (self.gender == "M")
    ###
    def is_female(self):
        return (self.gender == "F")

###########################################

class Battler(Character):
    """
    Class for characters involved in battles.
    Attributes : 
    - hp: health points
    - sp: special points
    - hpmax
    - spmax
    - status
    - skills
    """
    def __init__(self,
                 hp=0,
                 sp=0,
                 hpmax=0,
                 spmax=0,
                 force=0,
                 speed=0,
                 precision=0,
                 defense=0,
                 status=None,
                 skills=None):
        self.hp = hp
        self.sp = sp
        self.force = force
        self.speed = speed
        self.precision = precision
        self.defense = defense
        self.status = status
        self.skills = skills
        return
    ####
    def gain_hp(self, delta_hp=0, verbose=True):
        # no hp gain
        if delta_hp == 0:
            return True

        # negative hp gains are handled in 'loose_hp' function
        if delta_hp < 0:
            return self.loose_hp(-delta_hp, verbose)
        
        if self.hp == self.hpmax:
            # hp is already at max value
            return False
        else:
            # hp is increased
            delta_hp = min(self.hpmax - self.hp, delta_hp)
            self.hp += delta_hp
            if verbose:
                print(self.name + " gains " + str(delta_hp) + "HP!")
            return True
    ####   
    def loose_hp(self, delta_hp=0, verbose=True):
        # no hp loss
        if delta_hp == 0:
            return True
        
        # negative hp losses are handled in 'gain_hp' function
        if delta_hp < 0:
            return self.gain_hp(-delta_hp, verbose)
        
        if self.hp <= 0:
            # hp is already at min value
            return False
        else:
            # hp is decreased
            delta_hp = min(self.hp, delta_hp)
            self.hp -= delta_hp
            if verbose:
                print(self.name + " looses " + str(delta_hp) + "HP!")
            # ! Check for possible death (hp <= 0) must be done in calling routine
            return True   
            
###########################################

class Tieable(Character):
    """
    Class for tieable characters.
    Attributes:
    - escape_xp : experience is escaping
    - bound
    """
    def __init__(self, escape_xp=0, bound=False):
        self.escape_xp = escape_xp
        self.bound = bound
        return

###########################################

class Rigger(Character):
    """
    Class for rigger characters.
    Attributes:
    - rigging_xp : experience is rigging
    """
    def __init__(self, rigging_xp=0):
        self.rigging_xp = rigging_xp
        return

###########################################

class Enemy(Battler):
    """
    Class for enemy characters (inherits from Battler class)
    Attributes:
    - loot (what the party gains when the enemy is defeated)
    """
    def __init__(self, loot=None):
        self.loot = loot
        return

###########################################

class Hero(Battler, Tieable, Rigger):
    """
    Class for hero characters.
    Attributes:
    - xp (experience)
    - lv (level)
    - lv_lut (hero level look-up-table (xp, skills learned, ...))
    - equipment (weapons, armors, accessories, restraints, gag ...)
    """
    def __init__(self, xp=0, lv=1, lv_lut=None):
        self.xp = xp
        self.lv = lv
        self.lv_lut = lv_lut
        return
    ####
    def gain_xp(self, delta_xp=0, verbose=True):
        # negative xp gains are not allowed
        if delta_xp < 0:
            return False

        # no gain of xp
        if delta_xp == 0:
            return True

        # gain xp while incrementing levels
        if verbose:
            print(self.name + " gains " + str(delta_xp) + "XP!")
        remaining_xp = delta_xp
        while True:
            xp_to_next_lv = self.lv_lut.get_xp_to_next_lv(self.lv, self.xp)
            if xp_to_next_lv is None:
                # Hero
                self.xp += delta_xp
                return True
            else:
                if xp_to_next_lv > remaining_xp:
                    self.xp += remaining_xp
                    return True
                else:
                    self.xp += xp_to_next_lv
                    remaining_xp -= xp_to_next_lv
                    stat = self.level_up(verbose=True)       
    ####
    def level_up(self, verbose=True):
        if self.lv < len(self.lv_lut.levels):
            new_lv = self.lv_lut.levels[self.lv]
            self.lv += 1
            # print level-up report
            if verbose:
                print(self.name + " reaches level " + str(self.lv) + "!")
                if new_lv.delta_hpmax > 0:
                    print("HP max: " + str(self.hpmax) +  " -> " + str(self.hpmax + new_lv.delta_hpmax))
                if new_lv.delta_spmax > 0:
                    print("SP max: " + str(self.spmax) +  " -> " + str(self.spmax + new_lv.delta_spmax))
            # increase hp max value
            self.hpmax += new_lv.delta_hpmax
            
            # increase sp max value
            self.spmax += new_lv.delta_spmax

            # learn the new skills
            if new_lv.new_skills is not None:
                for skill in new_lv.new_skills:
                    # put following lines in a function 'learn_skill' >>>---
                    if skill not in self.skills:
                        if verbose:
                            print(self.name + " learns the skill '" +  skill.name + "'!")
                        self.skills.append(skill)
                    # ---<<<
            return True
        else:
            return False
    ###
    def __repr__(self):
        txt = "\n+======= HERO =======\n"
        txt += "| " + self.name + "\n"
        txt += "| Lv " + str(self.lv) + "\n"
        txt += "| HP = " + str(self.hp) + "/" + str(self.hpmax) + "\n"
        txt += "| SP = " + str(self.sp) + "/" + str(self.spmax) + "\n"
        xp_to_next_lv = self.lv_lut.get_xp_to_next_lv(self.lv, self.xp)
        txt += "| XP = " + str(self.xp)
        if xp_to_next_lv is not None:
            txt += "   (next Lv: " + str(xp_to_next_lv) + ")\n"
        else:
            txt += "\n"
        if self.status is not None:
            txt += "| ["
            nstatus = len(self.status) - 1
            for i, status in enumerate(self.status):
                txt += status.name
                if i < nstatus:
                    txt += ", "
                else:
                    txt += "]\n"
        txt += "+====================\n"
        return txt
###########################################

class Party:
    """
    A party is a team of heroes.
    Gathers shared attributes:
    - gold
    """
    def __init__(self, members=None, gold=0, inventary=None):
        self.members = members
        self.gold = gold
        self.inventary = inventary
        return
    ####
    def join(self, hero=None, verbose=True):
        """
        Make a hero join a party.
        """
        assert (isinstance(hero, Hero)), "Only heroes can join a party!"
        
        if hero not in self.members:
            # the hero joins the party successfully
            self.members.append(hero)
            if verbose:
                print(hero.name + " joins the party!")
            return True
        else:
            # the hero is already a member of the party
            return False
    ####
    def gain_gold(self, delta_gold=0, verbose=True):
        if delta_gold < 0:
            return loose_gold(self, -delta_gold, verbose)
        elif delta_gold > 0:
            self.gold += delta_gold
            if verbose:
                print("Party gains " + str(delta_gold) + "G!")
            return True
    ####
    def loose_gold(self, delta_gold=0, verbose=True):
        if delta_gold < 0:
            return gain_gold(self, -delta_gold, verbose)
        elif delta_gold > 0:
            self.gold -= delta_gold
            if verbose:
                print("Party looses " + str(delta_gold) + "G!")
            return True
    ####
    def add_to_inventary(self, item, verbose=True):
        if verbose:
            print("Party finds '" + item.name + "'!")
        self.inventary.items.append(item)
        return True
###########################################

class Level:
    """
    A class for levels.
    Attributes:
    - xp (xp amount required to reach this level)
    - delta_hpmax (hpmax value increment when this level is reached)
    - delta_spmax (spmax value increment when this level is reached)
    - new_skills (list of new skills learned when this level is reached)
    """
    def __init__(self, xp=0, delta_hpmax=0, delta_spmax=0, new_skills=None):
        self.xp = xp
        self.delta_hpmax = delta_hpmax
        self.delta_spmax = delta_spmax
        self.new_skills = new_skills
        return
###########################################

class LevelLUT:
    """
    A class for hero level look-up-tables.
    (basically an array of Level instances)
    Attributes : 
    - levels
    """
    def __init__(self, levels=None):
        self.levels = levels
        return
    ####
    def get_xp_to_next_lv(self, current_lv, current_xp):
        """
        Get amount of xp required to level up.
        """
        if current_lv >= len(self.levels):
            return None
        else:
            # levels start from 1, indices from 0, 
            # so the current level has index = current_lv - 1,
            # and next level has index current_lv
            return self.levels[current_lv].xp - current_xp
        
    
