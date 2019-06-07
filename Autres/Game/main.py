import character, battle, skill, status, item

lv_lut = character.LevelLUT(levels=[
    character.Level(xp=0,
                    delta_hpmax=0,
                    delta_spmax=0,
                    new_skills=[]),
    character.Level(xp=100,
                    delta_hpmax=5,
                    delta_spmax=4,
                    new_skills=[skill.Skill(name="Fire")])
    ])

for level in lv_lut.levels:
    for skill in level.new_skills:
        print(skill.name)


hero = character.Hero()
# Character properties
hero.name = "Asa"
hero.gender = "F"
# Battler properties
hero.hpmax = 20
hero.spmax = 10
hero.hp = hero.hpmax
hero.sp = hero.spmax
hero.force = 10
hero.speed = 10
hero.precision = 10
hero.defense = 10
hero.status = [
    status.Status(name="Asleep"),
    status.Status(name="Poison")]
hero.skills = []
# Hero properties
hero.xp = 0
hero.lv = 1
hero.lv_lut = lv_lut
# Tieable properties
hero.escape_xp = 0
hero.bound = False
# Rigger properties
hero.rigging_xp = 0



party = character.Party(members=[],
                        gold=100,
                        inventary=item.Inventary(items=[], weightmax=100))


ring = item.Equipable()
ring.name = "Ring of Power"
ring.use_in_battle = False
ring.weight = 0
ring.nb_use_max = -1
ring.delta_force = 100
ring.delta_speed = 100
ring.delta_precision = 100
ring.delta_defense = 100


loot = battle.Loot(xp=120,
                   gold=40,
                   items=[ring])


stat = party.join(hero)
for h in party.members:
    print(h)

#hero.gain_xp(120)
stat = battle.assign_loot(party, loot)


hero.gain_hp(-18)

for h in party.members:
    print(h)
    
hero.gain_xp(1000)
