from cacgan.data import AtmoStructureGroup

if __name__ == '__main__':
    mg = AtmoStructureGroup.from_csv()
    amine_gs = mg.group_by("amine")
    amine_gs = sorted(amine_gs, key=lambda x: len(x), reverse=True)
    namines = len(amine_gs)
    namines_unpopular = 0
    for g in amine_gs:
        print(g.first_amine, len(g))
        if len(g) < 5:
            namines_unpopular += 1
    print(namines, namines_unpopular)
