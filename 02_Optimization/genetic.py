from chromosome import Chromosome
from functools import cmp_to_key
from random import randint


def fitness(population):
    print("Fitness")
    population = sorted(population, key=cmp_to_key(Chromosome.comparator))
    population = [x for x in population if x.score > 0]
    return population


def crossover(population):
    total = len(population)
    new_population = []

    while total > 3:
        dad = population.pop(randint(0, total - 1))
        mom = population.pop(randint(0, total - 2))
        total -= 2

        sp1 = Chromosome([mom.angles[0], mom.angles[1], dad.angles[2], dad.angles[3], dad.angles[4], dad.angles[5]])
        sp2 = Chromosome([dad.angles[0], dad.angles[1], mom.angles[2], mom.angles[3], dad.angles[4], dad.angles[5]])
        sp3 = Chromosome([dad.angles[0], dad.angles[1], dad.angles[2], dad.angles[3], mom.angles[4], mom.angles[5]])
        sp4 = Chromosome([mom.angles[0], mom.angles[1], mom.angles[2], mom.angles[3], dad.angles[4], dad.angles[5]])
        sp5 = Chromosome([mom.angles[0], mom.angles[1], dad.angles[2], dad.angles[3], mom.angles[4], mom.angles[5]])
        sp6 = Chromosome([dad.angles[0], dad.angles[1], mom.angles[2], mom.angles[3], mom.angles[4], mom.angles[5]])

        new_population.append(sp1)
        new_population.append(sp2)
        new_population.append(sp3)
        new_population.append(sp4)
        new_population.append(sp5)
        new_population.append(sp6)

    return new_population


def main():
    population = []

    for x in range(0, 1000):
        specimen = Chromosome([randint(0, 360), randint(0, 360), randint(0, 360),
                               randint(0, 360), randint(0, 360), randint(0, 360)])

        population.append(specimen)

    age = 0
    while True:
        age += 1

        population = fitness(population)
        if age % 2 == 0:
            population = population[:min(5000, int(len(population) * 0.6))]

        new_population = crossover(population[:int(len(population) * 0.8)])

        the_best = population[0]
        print("Age: " + str(age))
        print(the_best)
        print("---------------------------------------------------------------")
        population = population + new_population


if __name__ == "__main__":
    main()
