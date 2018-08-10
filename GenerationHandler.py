from math import sqrt, factorial, ceil
from random import random, randrange, sample
import matplotlib.pyplot
import matplotlib.animation
import time

class GenerationHandler:

    def __init__(self, points, generation_size=20, number_of_generations=20, debug=True):
        self.points = points
        self.generation_size = generation_size
        self.number_of_generations = number_of_generations
        self.debug = debug
        self.current_generation = 0
        self.generations = {}
        self.generation = []
        self.all_solutions = []
        self.best_solution = [0, 0, 0]
        self.heuristics = {}
        self.generate_heuristics()
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

    def generate_heuristics(self):
        self.heuristics['relationships'] = {}
        self.heuristics['prefer'] = {}
        self.heuristics['avoid'] = {}
        self.heuristics['boundpair'] = []
        for c in range(len(self.points)):
            relationships = self.points[:]
            relationships.remove(self.points[c])
            relationships.sort(key=lambda x: self.fitness([self.points[c], x]))
            self.heuristics['relationships'][self.points[c]] = relationships[:]
            self.heuristics['prefer'][self.points[c]] = relationships[0:min(5, len(self.points) // 2)]
            self.heuristics['avoid'][self.points[c]] = relationships[-min(5, len(self.points) // 2):]
        skip = []
        for c in self.points:
            for o in self.heuristics['prefer'][c]:
                if c in self.heuristics['avoid'][o]:
                    self.heuristics['avoid'][o].remove(c)
            if c in skip:
                continue
            if self.heuristics['prefer'][self.heuristics['prefer'][c][0]][0] == c:
                self.heuristics['boundpair'].append((c, self.heuristics['prefer'][c][0]))
                skip.append(self.heuristics['prefer'][c][0])
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
        #if self.current_generation > 9 and self.generations[self.current_generation]['apexfitness'] < self.generations[self.current_generation - 10]['apexfitness']:
        #    selection[0][0] = self.generations[self.current_generation - 10]['apexstrand']
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
        for i in range(len(solution) - group_size + 1):
            groups.append(solution[i:i+group_size])
        groups.sort(key=lambda x: self.fitness(x))
        return groups[rank]

    def mutate(self, variant, solution):
        # variants: 0:single_swap, 1:group_swap, 2:cycle, 3:split_worst, 4:full_shuffle
        s = solution[:]
        if variant == 0:
            first = randrange(0, len(s))
            second = first
            while second == first:
                second = randrange(0, len(s))
            s[first], s[second] = s[second], s[first]
        if variant == 1:
            group_size = randrange(2, len(s) // 2 + 1)
            start = randrange(0, len(s) - 2 * group_size + 1)
            first = s[start:start + group_size]
            second = self.subgroup(s[start+group_size:], group_size, sample(range(0, len(s[start + group_size:]) - group_size + 1), k=1)[0])
            sstart = s.index(second[0])
            for i in range(0, group_size):
                s[start + i], s[sstart + i] = s[sstart + i], s[start + i]
        if variant == 2:
            for c in range(1, len(s)):
                s[c] = solution[c-1]
            s[0] = solution[-1]
            if s in self.all_solutions:
                s = self.mutate(0, s)
        return s

    def crossover(self, variant, f, m):
        # variants: 0:random_offset, 1:similar_groups, 2:best_groups, 3:avoid_similar
        child = []
        if variant == 0:
            length = randrange(1, len(f))
            start = randrange(0, len(f) - length + 1)
            child = f[start:start+length]
            for c in m:
                if c not in child:
                    child.append(c)
            return child
        pass

    def handle_mutations(self, mutation_map):
        next_generation = []
        for s in mutation_map[5]:
            mutation_map[4].append(self.mutate(0, s))
        for s in mutation_map[4]:
            mutation_map[2].append(self.mutate(0, s))
        for s in mutation_map[3]:
            mutation_map[1].append(self.mutate(0, s))
        for s in mutation_map[2]:
            mutation_map[0].append(self.mutate(0, s))
        for s in mutation_map[1]:
            mutation_map[0].append(self.mutate(ceil(random() * 3), s))
        for s in mutation_map[0]:
            next_generation.append(s)
        return next_generation

    def handle_crossovers(self, crossover_pool, crossover_map, mutation_map):
        while sum(len(l) for l in mutation_map.values()) < self.generation_size:
            roulette = sorted(list(crossover_map.keys()), key=lambda x: random() * crossover_map[x], reverse=True)
            mutation_map[0].append(self.crossover(0, crossover_pool[roulette[0]], crossover_pool[roulette[1]]))
        return mutation_map

    def next_generation(self, selection):
        mutation_map = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}
        crossover_pool = []
        crossover_map = {}
        crossover_index = 0
        mutation_map[1].append(selection[0][0])
        crossover_pool.append(selection[0][0])
        crossover_map[crossover_index] = 2.2
        crossover_index += 1
        for s in selection[1]:
            mutation_map[2].append(s)
            crossover_pool.append(s)
            crossover_map[crossover_index] = 0.8
            crossover_index += 1
        for s in selection[2]:
            mutation_map[3].append(s)
            crossover_pool.append(s)
            crossover_map[crossover_index] = 0.5
            crossover_index += 1
        for s in selection[3]:
            mutation_map[4].append(s)
            crossover_pool.append(s)
            crossover_map[crossover_index] = 0.25
            crossover_index += 1
        for s in selection[4]:
            mutation_map[5].append(s)
        mutation_map = self.handle_crossovers(crossover_pool, crossover_map, mutation_map)
        while sum(len(l) for l in mutation_map.values()) < self.generation_size:
            new_solution = sample(self.points, k=len(self.points))
            if new_solution not in self.all_solutions:
                mutation_map[0].append(new_solution)
                self.all_solutions.append(new_solution)
        return self.handle_mutations(mutation_map)

    def evolve(self):
        while self.current_generation < self.number_of_generations:
            self.evaluate()
            if self.best_solution == [0, 0, 0] or self.fitness(self.generation[0]) < self.best_solution[1]:
                self.best_solution[0] = self.generation[0]
                self.best_solution[1] = self.fitness(self.generation[0])
                self.best_solution[2] = self.current_generation
            self.generations[self.current_generation] = {}
            self.generations[self.current_generation]['genomes'] = self.generation
            self.generations[self.current_generation]['apexstrand'] = self.generation[0]
            self.generations[self.current_generation]['apexfitness'] = self.fitness(self.generation[0])
            self.generations[self.current_generation]['globalapex'] = self.best_solution[0]
            self.stats()
            selection = self.select()
            self.generation = self.next_generation(selection)
            self.current_generation += 1

    def animate(self, i):
        self.axis.clear()
        x, y = [c[0] for c in self.generations[i]['globalapex']], [c[1] for c in self.generations[i]['globalapex']]
        [self.axis.plot([x[p],x[p+1]], [y[p],y[p+1]], 'o-') for p in range(len(x) - 1)]

    def animation(self):
        self.fig = matplotlib.pyplot.figure()
        self.axis = self.fig.add_subplot(1,1,1)
        frames = [0]
        for g in range(self.current_generation):
            if self.generations[g]['globalapex'] != self.generations[frames[-1]]['globalapex']:
                frames.append(g)
        anim = matplotlib.animation.FuncAnimation(self.fig, self.animate, frames=frames, interval=50, repeat_delay=2500)
        matplotlib.pyplot.show()

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
            self.animation()

if __name__ == "__main__":
    #gh = GenerationHandler(points=[(1,1),(1,3),(3,5),(8,5),(12,2),(12,1),(10,0)], generation_size=20, number_of_generations=10, debug=False)
    #gh = GenerationHandler(points=[(1,1),(5,4),(5,7),(5,16),(6,11),(6,21),(10,23),(15,23),(16,10),(16,12),(18,9),(18,17),(19,20),(20,14)], generation_size=20, number_of_generations=400, debug=False)
    gh = GenerationHandler(points=[(4,2),(8,4),(19,2),(4,12),(9,9),(10,10),(1,1),(4,9),(10,14),(16,16),(13,6),(5,4),(5,7),(5,16),(6,11),(6,21),(10,23),(15,23),(16,5),(16,12),(18,9),(18,17),(19,20),(20,14)], generation_size=20, number_of_generations=15000, debug=False)
