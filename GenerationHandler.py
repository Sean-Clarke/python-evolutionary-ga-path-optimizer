from math import sqrt, factorial
from random import random, randrange, sample

class GenerationHandler:

    def __init__(self, points, generation_size=20, number_of_generations=20):
        self.generation_size = generation_size
        self.number_of_generations = number_of_generations
        self.points = points
        self.current_generation = 0
        self.generations = {}
        self.generation = []
        self.all_solutions = []
        self.best_solution = [0, 0, 0]
        self.initialize()
        self.evolve()
        self.stats()

    def initialize(self):
        if self.generation_size > factorial(len(self.points)):
            raise ValueError("generation_size must not be larger than factorial(len(points))")
        if self.generation_size != 20:
            raise ValueError("generation_size is currently only implemented to have the value 20")
        while len(self.generation) < self.generation_size:
            new_solution = sample(self.points, k=len(self.points))
            if new_solution not in self.all_solutions:
                self.generation.append(new_solution)
                self.all_solutions.append(new_solution)
        return True

    def fitness(self, s):
        f = 0
        for c in range(len(s) - 1):
            f += sqrt((s[c][0] - s[c + 1][0])**2 + (s[c][1] - s[c+1][1])**2)
        return f

    def evaluate(self):
        self.generation.sort(key=lambda x: self.fitness(x))

    def select(self):
        selection = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}
        selection[0].append(self.generation[0])
        for s in self.generation[1:4]:
            if random() > 0.2:
                selection[1].append(s)
        for s in self.generation[4:8]:
            if random() > 0.4:
                selection[2].append(s)
        for s in self.generation[8:12]:
            if random() > 0.6:
                selection[3].append(s)
        for s in self.generation[12:16]:
            if random() > 0.8:
                selection[4].append(s)
        for s in self.generation[16:20]:
            selection[5].append(s)
        return selection

    def subgroup(self, solution, group_size, rank=0):
        """Given a solution, an int group_size of points, and an int fitness rank, returns the rank fittest group of points of size three within the solution"""
        groups = []
        for i in range(len(solution)-group_size):
            groups.append(solution[i:i+group_size])
        groups.sort(key=lambda x: self.fitness(x))
        return groups[rank]

    def mutate(self, variant, solution):
        # variants: 0:single_swap, 1:group_swap, 2:cycle, 3:split_worst, 4:full_shuffle
        if variant == 0:
            first = randrange(0, len(solution))
            second = first
            while second == first:
                second = randrange(0, len(solution))
            solution[first], solution[second] = solution[second], solution[first]
        return solution

    def crossover(self, variant, *parent_solutions):
        # variants: 0:random_offset, 1:similar_groups, 2:best_groups, 3:avoid_similar
        child = []
        if variant == 0:
            length = randrange(1, len(parent_solutions[0]))
            start = randrange(0, len(parent_solutions[0]) - length + 1)
            print(start)
            print(length)
            child = parent_solutions[0][start:start+length]
            return child
        pass

    def handle_mutations(self, mutation_map):
        next_generation = []
        for s in mutation_map[5]:
            mutation_map[4].append(self.mutate(0, s))
        for s in mutation_map[4]:
            mutation_map[3].append(self.mutate(0, s))
        for s in mutation_map[3]:
            mutation_map[2].append(self.mutate(0, s))
        for s in mutation_map[2]:
            mutation_map[1].append(self.mutate(0, s))
        for s in mutation_map[1]:
            mutation_map[0].append(self.mutate(0, s))
        for s in mutation_map[0]:
            next_generation.append(s)
        return next_generation

    def handle_crossovers(self, crossover_pool, mutation_map):
        while sum(len(l) for l in mutation_map.values()) < self.generation_size and len(crossover_pool) > 1:
            pass
        return mutation_map

    def next_generation(self, selection):
        mutation_map = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}
        crossover_pool = []

        mutation_map[1].append(selection[0][0])
        for s in selection[1]:
            mutation_map[2].append(s)
        for s in selection[2]:
            mutation_map[3].append(s)
        for s in selection[3]:
            mutation_map[4].append(s)
        for s in selection[4]:
            mutation_map[5].append(s)

        #split selection into mutation_map and crossover_pool
        mutation_map = self.handle_crossovers(crossover_pool, mutation_map)
        while sum(len(l) for l in mutation_map.values()) < self.generation_size:
            new_solution = sample(self.points, k=len(self.points))
            if new_solution not in self.all_solutions:
                mutation_map[0].append(new_solution)
                self.all_solutions.append(new_solution)
        next_generation = self.handle_mutations(mutation_map)
        return next_generation

    def evolve(self):
        while self.current_generation < self.number_of_generations:
            self.evaluate()
            self.generations[self.current_generation] = self.generation
            if self.best_solution == [0, 0, 0] or self.fitness(self.generation[0]) < self.best_solution[1]:
                self.best_solution[0] = self.generation[0]
                self.best_solution[1] = self.fitness(self.generation[0])
                self.best_solution[2] = self.current_generation
            self.stats()
            selection = self.select()
            generation = self.next_generation(selection)
            self.current_generation += 1
    def stats(self):
        if self.current_generation != self.number_of_generations:
            print('Generation ' + str(self.current_generation) + ':')
            print('   Fittest Solution: ' + str(self.generation[0]))
            print('   Best Fitness: ' + str(self.fitness(self.generation[0])))
            print('   Average Fitness: ' + str(sum(self.fitness(s) for s in self.generation) / len(self.generation)) + '\n')
        else:
            print('Final Stats:')
            print('   Fittest solution found in generation ' + str(self.best_solution[2]))
            print('   Fittest Solution: ' + str(self.best_solution[0]))
            print('   Fittest Solution Fitness: ' + str(self.best_solution[1]))

if __name__ == "__main__":
    gh = GenerationHandler(points=[(5,9),(8,6),(8,2),(5,0),(1,0),(0,2),(0,3),(1,9)], generation_size=20, number_of_generations=100)
