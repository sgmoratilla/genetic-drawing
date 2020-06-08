import cv2
import numpy as np
import random

from deap import base
from deap import creator 
from deap import tools
from deap.tools.support import HallOfFame

from drawer import *

import logging as logging

logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s')
log = logging.getLogger("genetic-drawing")
#log.setLevel(level=logging.DEBUG)

class DrawingRestrictions:

    def __init__(self, image_bounds, image_gradient, brush_range_limits, n_brushes):
        self.image_bounds = image_bounds
        self.image_gradient = image_gradient
        self.brush_range_limits = brush_range_limits
        self.n_brushes = n_brushes
                      
        #IMG GRADIENT
        self.imageMag = image_gradient[0]
        self.imageAngles = image_gradient[1]

    def calculate_brush_size(self, stage, total_stages):
        return [self._calculate_brush_size(self.brush_range_limits[0], stage, total_stages), self._calculate_brush_size(self.brush_range_limits[1], stage, total_stages)]

    def _calculate_brush_size(self, brange, stage, total_stages):
        bmin = brange[0]
        bmax = brange[1]
        t = stage / max(total_stages - 1, 1)
        return (bmax - bmin) * (-t * t + 1) + bmin


class DrawingProblem:
    def __init__(self, image_path, sampling_mask=None):
        self.original_image = cv2.imread(image_path)
        self.image_greyscale = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.image_shape = self.image_greyscale.shape
        self.image_gradients = image_gradient(self.image_greyscale)
        self.sampling_mask = sampling_mask

        self.restrictions = DrawingRestrictions(self.image_shape, self.image_gradients, brush_range_limits = [[0.1, 0.3], [0.3, 0.7]], n_brushes = 4)
        self.drawer = ImageDrawer(self.image_shape)
        self.target_image = self.image_greyscale

    def set_sampling_mask(image_path):
        self.sampling_mask = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
        
    def gen_new_positions(self):
        if self.sampling_mask is not None:
            pos = sample_from_img(self.sampling_mask)
            posY = pos[0][0]
            posX = pos[1][0]
        else:
            posY = int(random.randrange(0, self.image_shape[0]))
            posX = int(random.randrange(0, self.image_shape[1]))
        
        return [posY, posX]

    def create_sampling_mask(self, s, stages):
        percent = 0.2
        start_stage = int(stages * percent)
        sampling_mask = None
        if s >= start_stage:
            t = (1.0 - (s - start_stage) / max(stages - start_stage - 1, 1)) * 0.25 + 0.005
            sampling_mask = self.calc_sampling_mask(t)
        return sampling_mask

    def calc_sampling_mask(self, blur_percent):
        img = np.copy(self.img_grey)
        # Calculate gradient 
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
        # Python Calculate gradient magnitude and direction ( in degrees ) 
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        #calculate blur level
        w = img.shape[0] * blur_percent
        if w > 1:
            mag = cv2.GaussianBlur(mag,(0,0), w, cv2.BORDER_DEFAULT)
        #ensure range from 0-255 (mostly for visual debugging, since in sampling we will renormalize it anyway)
        scale = 255.0 / mag.max()
        return mag * scale

    def error(self, proposed_image):
        return self.__error(proposed_image, self.target_image)
        
    def __error(self, proposed_image, target_image):
        # calc fitness only in the ROI
        diff1 = cv2.subtract(target_image, proposed_image) #values are too low
        diff2 = cv2.subtract(proposed_image, target_image) #values are too high
        totalDiff = cv2.add(diff1, diff2)
        totalDiff = np.sum(totalDiff)
        return totalDiff, proposed_image

class DrawingIndividual:

    def __init__(self, n_strokes, brush_range):
        self.n_strokes = n_strokes
        self.fenotype = []

        self.minSize = brush_range[0] #0.1 #0.3
        self.maxSize = brush_range[1] #0.3 # 0.7
        self.brushSide = 300 # brush image resolution in pixels
        self.padding = int(self.brushSide * self.maxSize / 2 + 5)

     
    def init_random(self, drawing_problem):
        restrictions = drawing_problem.restrictions

        # initialize random fenotype
        for i in range(self.n_strokes):
            # random color
            color = random.randrange(0, 255)
            size = random.random() * (self.maxSize - self.minSize) + self.minSize
            # random pos
            posY, posX = drawing_problem.gen_new_positions()
            # random rotation
            '''
            start with the angle from image gradient
            based on magnitude of that angle direction, adjust the random angle offset.
            So in places of high magnitude, we are more likely to follow the angle with our brushstroke.
            In places of low magnitude, we can have a more random brushstroke direction.
            '''
            localMag = restrictions.imageMag[posY][posX]
            localAngle = restrictions.imageAngles[posY][posX] + 90 # perpendicular to the dir
            rotation = random.randrange(-180, 180) * (1 - localMag) + localAngle
            # random brush number
            brushNumber = random.randrange(1, restrictions.n_brushes)
            # append data
            self.fenotype.append(Stroke(color, posY, posX, size, rotation, brushNumber))


    def mutate(self, drawing_problem):
        for i in range(len(self.fenotype)):
            # TODO: don't mutate every gen
            self.__mutate(drawing_problem, i)


    def __mutate(self, drawing_problem, index):
        restrictions = drawing_problem.restrictions

        #create a copy of the list and get its child  
        fenotype_copy = np.copy(self.fenotype)           
        child = fenotype_copy[index]

        indexOptions = [0,1,2,3,4,5]
        changeIndices = []
        changeCount = random.randrange(1, len(indexOptions)+1)
        for i in range(changeCount):
            indexToTake = random.randrange(0, len(indexOptions))
            changeIndices.append(indexOptions.pop(indexToTake))

        #mutate selected items
        np.sort(changeIndices)
        changeIndices[:] = changeIndices[::-1]
        for changeIndex in changeIndices:
            if changeIndex == 0:# if color
                child.color = int(random.randrange(0, 255))
                log.debug(f'new color: {child.color}')
            elif changeIndex == 1 or changeIndex == 2:#if pos Y or X
                child.posY, child.posX = drawing_problem.gen_new_positions()
                log.debug(f'new posY: {child.posY} | new posX: {child.posX}')
            elif changeIndex == 3: #if size
                child.size = random.random() * (self.maxSize - self.minSize) + self.minSize
                log.debug(f'new size: {child.size}')
            elif changeIndex == 4: #if rotation
                log.debug(f'trying to mutate rotation with posY={child.posY}, posX={child.posX}')
                localMag = restrictions.imageMag[int(child.posY)][int(child.posX)]
                localAngle = restrictions.imageAngles[int(child.posY)][int(child.posX)] + 90 #perpendicular
                child.rotation = random.randrange(-180, 180) * (1 - localMag) + localAngle
                log.debug(f'new rotation: {child.rotation}')
            elif changeIndex == 5: #if  brush number
                child.brushNumber = random.randrange(1, restrictions.n_brushes)
                #print('new brush: ', child[5])

class NotebookDrawingMonitor:

    def __init__(self, drawer, show_progress_images=True):
        self.show_progress_images = show_progress_images
        self.hall_of_fame = HallOfFame(5) # Saving top 5
        self.drawer = drawer

        # start with an empty black img
        self.image_buffer = [np.zeros((drawer.image_shape[0], drawer.image_shape[1]), np.uint8)]

    def submit(self, population):
        from IPython.display import clear_output
        import matplotlib.pyplot as plt

        self.hall_of_fame.update(population)
        fittest_image = self.drawer.draw(self.hall_of_fame[0].fenotype, self.hall_of_fame[0].padding)

        self.image_buffer.append(fittest_image)
    
        clear_output(wait=True)

        if self.show_progress_images is True:
            #plt.imshow(sampling_mask, cmap='gray')
            plt.imshow(fittest_image, cmap='gray')
            plt.show()

class GeneticDrawing:

    def __init__(self, drawing_problem, seed):
        log.info("Initializing GeneticDrawing")

        self.drawing_problem = drawing_problem

        random.seed(seed)

        # No crossover yet, not implemented
        self.crossover_prob = 0
        # Original algorithm mutated always
        self.mutation_prob = 1

        self.toolbox = base.Toolbox()

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", DrawingIndividual, fitness=creator.FitnessMin)

        # Custom initializer
        self.toolbox.register("individual", deap_init_individual, creator.Individual)
        # DEAP standard initRepeat
        self.restrictions = DrawingRestrictions(image_bounds = drawing_problem.image_shape, image_gradient = drawing_problem.image_gradients, brush_range_limits = drawing_problem.restrictions.brush_range_limits, n_brushes = drawing_problem.restrictions.n_brushes)
        self.toolbox.register("population", deap_init_population, list, self.toolbox.individual, drawing_problem = self.drawing_problem)

        # Custom mutation function
        self.toolbox.register("select", deap_select_population)

        # Custom mutation function
        self.toolbox.register("mutate", deap_mutate_individual)

        # Custom fitness function
        self.toolbox.register("evaluate", deap_evaluate_individual, drawing_problem=drawing_problem)


    def generate(self, stages=100, n_generations=100, population_size=50, individual_size=10, monitor=None):
        log.info(f'Starting working. {stages} stages, {n_generations} n_genrations, {population_size} population_size, {individual_size} individual_size')

        toolbox = self.toolbox
        for s in range(stages):
            log.debug(f'Starting stage {s}')

            brush_range = self.restrictions.calculate_brush_size(s, stages) 
            population = self.toolbox.population(size=population_size, individual_size=individual_size, brush_range=brush_range)

            # Evaluate the entire population
            fitnesses = map(toolbox.evaluate, population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            for g in range(n_generations):
                log.debug(f'Running stage {s} generation {g}')

                # Select the next generation individuals
                offspring = toolbox.select(population, len(population))
                log.debug(f's{s}-g{g} offspring {offspring}')

                # Clone the selected individuals
                offspring = [toolbox.clone(o) for o in offspring]
                #offspring = map(toolbox.clone, offspring)
                log.debug(f's{s}-g{g} clone {offspring}')

                if (self.crossover_prob > 0):
                    # Apply crossover and mutation on the offspring
                    for child1, child2 in zip(offspring[::2], offspring[1::2]):
                        if random.random() < crossover_prob:
                            toolbox.mate(child1, child2)
                            del child1.fitness.values
                            del child2.fitness.values
                    log.debug(f's{s}-g{g} crossover {offspring}')

                if (self.mutation_prob > 0):
                    for mutant in offspring:
                        if random.random() < self.mutation_prob:
                            toolbox.mutate(mutant, self.drawing_problem)
                            del mutant.fitness.values
                    log.debug(f's{s}-g{g} mutation {offspring}')

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # The population is entirely replaced by the offspring
                population[:] = offspring

                log.debug(f's{s}-g{g} final population {offspring}')
                if monitor is not None:
                    monitor.submit(population)

            log.debug(f'Stage {s} ended')

        return population


def deap_init_individual(icls, size, brush_range, drawing_problem):
    individual = icls(n_strokes = size, brush_range = brush_range)
    individual.init_random(drawing_problem)
    return individual

def deap_init_population(pcls, init_individual, size, individual_size, brush_range, drawing_problem):
    return pcls(init_individual(individual_size, brush_range, drawing_problem) for i in range(size))

def deap_select_population(population, size):
    return population

def deap_mutate_individual(individual, drawing_problem):
    individual.mutate(drawing_problem)
    return individual

def deap_evaluate_individual(individual, drawing_problem):
    strokes = individual.fenotype
    padding = individual.padding
    proposal = drawing_problem.drawer.draw(strokes, padding)
    diff, image = drawing_problem.error(proposal)

    return diff,
