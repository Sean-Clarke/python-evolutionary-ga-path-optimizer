from math import sqrt, factorial, ceil
from random import random, randrange, sample, choices
import matplotlib.pyplot
import matplotlib.animation

class GenerationHandler:

    def __init__(self, points, generation_size=20, number_of_generations=20, answer=False, debug=True):
        self.points = points
        self.generation_size = generation_size
        self.number_of_generations = number_of_generations
        self.answer = answer
        self.debug = debug
        self.current_generation = 0
        self.generations = {}
        self.generation = []
        self.all_solutions = []
        self.best_solution = [0, 0, 0]
        self.heuristics = {}
        self.meta = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
        self.generate_heuristics()
        self.initialize()
        self.evolve()
        self.stats()

    def initialize(self):
        if self.generation_size > factorial(len(self.points)):
            raise ValueError("generation_size must not be larger than factorial(len(points))")
        if self.generation_size < 2:
            raise ValueError("generation_size must be greater than 2")
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
                self.heuristics['boundpair'].append([c, self.heuristics['prefer'][c][0]])
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
        if self.current_generation > 9 and self.generations[self.current_generation]['apexfitness'] < self.generations[self.current_generation - 10]['apexfitness']:
            selection[1].append(self.generations[self.current_generation - 10]['apexstrand'])
        selection[1].append(self.generation[1])
        for s in self.generation[2:max(len(self.generation) // 5, 3)]:
            if random() > 0.2:
                selection[1].append(s)
        for s in self.generation[max(len(self.generation) // 5, 3):max(2 * len(self.generation) // 5, 4)]:
            if random() > 0.4:
                selection[2].append(s)
        for s in self.generation[max(2 * len(self.generation) // 5, 4):max(3 * len(self.generation) // 5, 5)]:
            if random() > 0.6:
                selection[3].append(s)
        for s in self.generation[max(3 * len(self.generation) // 5, 5):max(4 * len(self.generation) // 5, 6)]:
            if random() > 0.8:
                selection[4].append(s)
        for s in self.generation[max(4 * len(self.generation) // 5, 6):max(len(self.generation), 7)]:
            selection[5].append(s)
        return selection

    def subgroup(self, solution, group_size, rank=0):
        """Given a solution, an int group_size of points, and an int fitness rank, returns the rank fittest group of points of size three within the solution"""
        groups = []
        for i in range(len(solution) - group_size + 1):
            groups.append(solution[i:i+group_size])
        groups.sort(key=lambda x: self.fitness(x))
        return groups[rank]

    def random_swap(self, solution, use_heuristics):
        exhausted = []
        used_pairs = []
        while True:
            s = solution[:]
            if len(exhausted) > len(s) - 2:
                return self.mutate(randrange(0, 9), s, use_heuristics)
            first = randrange(0, len(s))
            while first in exhausted:
                first = randrange(0, len(s))
            second = first
            while second == first or second in exhausted or set([first, second]) in used_pairs:
                second = randrange(0, len(s))
            s[first], s[second] = s[second], s[first]
            if use_heuristics:
                net = 0
                if first > 0:
                    if s[first - 1] in self.heuristics['avoid'][s[first]]:
                        net -= 1
                    if s[first - 1] in self.heuristics['prefer'][s[first]]:
                        net += 1
                if first < len(s) - 1:
                    if s[first + 1] in self.heuristics['avoid'][s[first]]:
                        net -= 1
                    if s[first + 1] in self.heuristics['prefer'][s[first]]:
                        net += 1
                if second > 0:
                    if s[second - 1] in self.heuristics['avoid'][s[second]]:
                        net -= 1
                    if s[second - 1] in self.heuristics['prefer'][s[second]]:
                        net += 1
                if second < len(s) - 1:
                    if s[second + 1] in self.heuristics['avoid'][s[second]]:
                        net -= 1
                    if s[second + 1] in self.heuristics['prefer'][s[second]]:
                        net += 1
                if net < 0:
                    used_pairs.append(set([first, second]))
                    first_exhaustion = 0
                    second_exhaustion = 0
                    for p in used_pairs:
                        if first in p:
                            first_exhaustion += 1
                        if second in p:
                            second_exhaustion += 1
                    if first_exhaustion > len(s) - 1:
                        exhausted.append(first)
                    if second_exhaustion > len(s) - 1:
                        exhausted.append(second)
                    continue
            return s
            
    def group_swap(self, s, use_heuristics):
        group_size = randrange(2, len(s) // 2 + 1)
        start = randrange(0, len(s) - 2 * group_size + 1)
        first = s[start:start + group_size]
        second = self.subgroup(s[start+group_size:], group_size, sample(range(0, len(s[start + group_size:]) - group_size + 1), k=1)[0])
        sstart = s.index(second[0])
        for i in range(0, group_size):
            s[start + i], s[sstart + i] = s[sstart + i], s[start + i]
        return s

    def cycle(self, solution, attempts):
        s = solution[:]
        for c in range(1, len(s)):
            s[c] = solution[c-1]
        s[0] = solution[-1]
        if s in self.all_solutions:
            if attempts < len(s):
                self.cycle(s, attempts + 1)
            else:
                s = self.mutate(randrange(0, 9), s)
        return s
        
    def smart_cycle(self, solution):
        s = solution[:]
        left, right = s[-1], s[0]
        f = self.fitness(s[-1:] + s[0:1])
        for i in range(0, len(s) - 1):
            if self.fitness(s[i:i+2]) > f:
                left, right, f = s[i], s[i+1], self.fitness(s[i:i+2])
        #s = self.mutate(randrange(0, 9), s)
        while not (s[0] == right and s[-1] == left):
            for c in range(1, len(s)):
                s[c] = solution[c-1]
            s[0] = solution[-1]
            solution = s[:]
        return s
        
    def flip_ends(self, solution):
        end = solution[0]
        for i in range(1, len(solution) - 1):
            if self.fitness([end, solution[i+1]]) < self.fitness(solution[i:i+2]):
                solution = list(reversed(solution[0:i+1])) + solution[i+1:]
                break
        end = solution[-1]
        for i in reversed(range(1, len(solution) - 1)):
            if self.fitness([end, solution[i-1]]) < self.fitness(solution[i-1:i+1]):
                solution = solution[0:i] + list(reversed(solution[i:]))
                break
        return solution

    def mutate(self, variant, solution, use_heuristics=True):
        # variants: 0:best_swap, 1:cycle_once, 2:improved_swap, 3:smart_cycle, 4:random_swap, 5:group_swap, 6:flip_ends, 7:split_worst, 8:full_shuffle
        self.meta[variant] += 1
        s = solution[:]
        
        if variant == 0: # best_swap
            """ """
            s = self.random_swap(s, use_heuristics)
            
        if variant == 1: # cycle_once
            """offsets the order of the coordinates in the solution until a unique solution is given"""
            s = self.cycle(s, 0)
            
        if variant == 2: # improved_swap
            """searches through the solution for any avoid pairs, then non-prefer pairs, swapping place to have better neighbours"""
            s = self.random_swap(s, use_heuristics)
            
        if variant == 3: # smart_cycle
            """offsets the order of coordinates until the least fir neighbouring pair are separated"""
            s = self.smart_cycle(s)
        
        if variant == 4: # random_swap
            """randomly chooses two coordinates to swap places in the solution"""
            s = self.random_swap(s, use_heuristics)

        if variant == 5: # group_swap
            """randomly chooses two groups of coordinates to swap places in the solution"""
            s = self.group_swap(s, use_heuristics)

        if variant == 6: # flip_ends
            """ """
            s = self.flip_ends(s)
                
        if variant == 7: # split_worst
            """"""
            pass
            
        if variant == 8: # full_shuffle
            """randomly shuffles the solution"""
            s = sample(self.points, k=len(self.points))
                
        return s

    def crossover(self, variant, f, m):
        # variants: 0:random_offset, 1:similar_groups, 2:best_groups, 3:avoid_similar
        if variant != 0:
            variant = 0
        child = []
        if variant == 0:
            length = randrange(1, len(f))
            start = randrange(0, len(f) - length + 1)
            child = f[start:start+length]
            for c in m:
                if c not in child:
                    child.append(c)
            if child == f or child == m:
                child = self.crossover(variant, f, m)
            return child

    def handle_mutations(self, mutation_map):
        next_generation = []
        for i in sorted(list(mutation_map.keys()), reverse=True):
            for s in mutation_map[i]:
                if i != 0:
                    mutation = False
                    while not mutation:
                        if i == 1:
                            v = randrange(0, 4)
                        else:
                            v = randrange(4-i, 5+i)
                        if i == 5:
                            v = 8
                        mutation = self.mutate(v, s)
                    mutation_map[max(i, 2) - 2].append(mutation)
                else:
                    while s in self.all_solutions:
                        s = self.mutate(randrange(0, 9), s)
                    self.all_solutions.append(s)
                    next_generation.append(s)
        return next_generation

    def handle_crossovers(self, crossover_pool, crossover_weights, mutation_map, allowance):
        starchild = self.crossover(0, crossover_pool[0], crossover_pool[1])
        if allowance > 0:
            mutation_map[0].append(starchild)
            allowance -= 1
        if allowance > 0:
            mutation_map[2].append(starchild)
            allowance -= 1
        while allowance > 0:
            f = choices(crossover_pool, weights=crossover_weights, k=1)[0]
            fw = crossover_weights[crossover_pool.index(f)] * 0.8
            crossover_weights[crossover_pool.index(f)] = 0
            m = choices(crossover_pool, weights=crossover_weights, k=1)[0]
            crossover_weights[crossover_pool.index(f)] = fw
            mutation_map[0].append(self.crossover(randrange(0, 4), f, m))
            allowance -= 1
        return mutation_map

    def next_generation(self, selection):
        mutation_map = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]} # 0:no mutate, 1:cycle or improve, 2: random, 3: random then 1, 4: broad random then 2, 5: full shuffle then 3
        crossover_pool = []
        crossover_weights = []
        mutation_map[1].append(selection[0][0])
        mutation_map[2].append(selection[0][0])
        mutation_map[3].append(selection[0][0])
        if self.generation_size > 17:
            mutation_map[1].append(selection[0][0])
            mutation_map[2].append(selection[0][0])
            mutation_map[4].append(selection[0][0])
        crossover_pool.append(selection[0][0])
        crossover_weights.append(20)
        for s in selection[1]:
            mutation_map[2].append(s)
            crossover_pool.append(s)
            crossover_weights.append(12)
        for s in selection[2]:
            mutation_map[3].append(s)
            crossover_pool.append(s)
            crossover_weights.append(8)
        for s in selection[3]:
            mutation_map[4].append(s)
            crossover_pool.append(s)
            crossover_weights.append(4)
        for s in selection[4]:
            mutation_map[5].append(s)
        mutation_map = self.handle_crossovers(crossover_pool, crossover_weights, mutation_map, self.generation_size - sum(len(l) for l in mutation_map.values()))
        return self.handle_mutations(mutation_map)

    def evolve(self):
        while self.current_generation < self.number_of_generations:
            self.evaluate()
            self.log()
            if round(self.best_solution[1], 10) == round(self.answer, 10):
                self.number_of_generations = self.current_generation + 1
            selection = self.select()
            self.generation = self.next_generation(selection)
            self.current_generation += 1

    def log(self):
        if self.best_solution == [0, 0, 0] or self.fitness(self.generation[0]) < self.best_solution[1]:
            self.best_solution[0] = self.generation[0]
            self.best_solution[1] = self.fitness(self.generation[0])
            self.best_solution[2] = self.current_generation
        self.generations[self.current_generation] = {}
        self.generations[self.current_generation]['genomes'] = self.generation
        self.generations[self.current_generation]['apexstrand'] = self.generation[0]
        self.generations[self.current_generation]['apexfitness'] = self.fitness(self.generation[0])
        self.generations[self.current_generation]['globalapex'] = self.best_solution[0]
        if self.debug:
            self.stats()
            input("Press any key")

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
        anim = matplotlib.animation.FuncAnimation(self.fig, self.animate, frames=frames, interval=50, repeat_delay=1500)
        matplotlib.pyplot.show()

    def stats(self):
        if self.current_generation != self.number_of_generations:
            print('Generation ' + str(self.current_generation) + ':')
            print('  Fittest Solution: ' + str(self.generation[0]))
            print('  Best Fitness: ' + str(self.fitness(self.generation[0])))
            print('  Average Fitness: ' + str(sum(self.fitness(s) for s in self.generation) / len(self.generation)) + '\n')
        else:
            print('Final Stats:')
            print('  Fittest solution found in generation ' + str(self.best_solution[2]))
            print('  Fittest Solution: ' + str(self.best_solution[0]))
            print('  Fittest Solution Fitness: ' + str(self.best_solution[1]))
            #for k,v in self.meta.items():
            #    print(str(k) + ': ' + str(v))
            self.animation()

if __name__ == "__main__":
    #curve = GenerationHandler(points=[(1,1),(3,1),(6,2),(10,4),(15,8),(21,16)], generation_size=20, number_of_generations=10, answer=26.037537852600806, debug=False)
    #circle = GenerationHandler(points=[(2,1),(1,3),(3,5),(5,0),(9,5),(8,0),(11,4),(11,1),(12,2)], generation_size=20, number_of_generations=20, answer=20.275399939955413, debug=False)
    spiral = GenerationHandler(points=[(25,26),(26,25),(27,25),(28,26),(28,28),(27,30),(25,31),(22,30),(20,28),(19,23),(20,19),(22,17),(25,16),(28,16),(31,18),(33,22),(34,27),(33,33),(30,39),(26,41),(22,41),(19,40)], generation_size=20, number_of_generations=300, answer=72.10618461080725, debug=False)
    #canada = GenerationHandler(points=[(2,2),(4,47),(5,6),(7,15),(8,41),(9,27),(12,17),(18,7),(18,14),(19,27),(22,43),(23,19),(27,23),(31,8),(33,3),(33,11),(37,1),(37,39),(38,11),(39,6),(42,2),(47,9),(48,26),(49,6),(49,12)], generation_size=20, number_of_generations=2000, debug=False) #answer may be 200.04912909882898
    #europe = GenerationHandler(points=[(4,87),(12,15),(16,54),(19,14),(19,38),(20,26),(21,51),(25,62),(26,15),(26,48),(29,36),(35,18),(39,28),(39,45),(41,13),(43,6),(46,21),(48,67),(50,55),(51,44),(52,38),(53,30),(55,62),(57,24),(57,29),(62,40),(63,24),(64,0),(67,70),(70,11),(70,47),(73,62),(79,35),(87,53)], generation_size=20, number_of_generations=2000, debug=False)
    pass
