import cv2
import os
import numpy as np
import random

from deap import base
from deap import creator 
from deap import tools
from deap.tools.support import HallOfFame

from drawer import *

from joblib import Parallel, delayed
from joblib import parallel_backend
import logging as logging

import matplotlib.pyplot as plt
from IPython.display import clear_output

logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s')
log = logging.getLogger("genetic-drawing")

class DrawingRestrictions:

    def __init__(self, image_bounds, image_gradient, brush_range, n_brushes):
        self.image_bounds = image_bounds
        self.image_gradient = image_gradient
        self.brush_range = brush_range
        self.n_brushes = n_brushes
                      
        #IMG GRADIENT
        self.imageMag = image_gradient[0]
        self.imageAngles = image_gradient[1]

class DrawingBrushesRange:
    def __init__(self, brush_range_limits = [[0.1, 0.3], [0.3, 0.7]]):
        self.brush_range_limits = brush_range_limits

    def calculate_brush_range(self, stage, total_stages):
        return [self._calculate_brush_size(self.brush_range_limits[0], stage, total_stages), self._calculate_brush_size(self.brush_range_limits[1], stage, total_stages)]

    def _calculate_brush_size(self, brange, stage, total_stages):
        bmin = brange[0]
        bmax = brange[1]
        t = stage / max(total_stages - 1, 1)
        return (bmax - bmin) * (-t * t + 1) + bmin


class DrawingProblem:
    def __init__(self, image_path = None, color_image = None, brush_range = [1, 1], sampling_mask = None, initial_drawer = None):
        if image_path is None and color_image is None:
            raise ValueError("Either image_path or color_image must be set")

        if image_path is not None and color_image is not None:
            raise ValueError("image_path and color_image cannot be set at once")

        if image_path is not None:
            if os.path.exists(image_path) and os.path.isfile(image_path):
                self.color_image = cv2.imread(image_path)
            else:
                raise ValueError(f"File {image_path} doesn't exist")
              
        if color_image is not None:
            self.color_image = color_image

        self.image_greyscale = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)
        self.image_shape = self.image_greyscale.shape
        self.image_gradients = image_gradient(self.image_greyscale)

        self.brush_range = brush_range
        self.sampling_mask = sampling_mask

        self.restrictions = DrawingRestrictions(self.image_shape, self.image_gradients, brush_range = brush_range, n_brushes = 4)
        if initial_drawer is None:
            self.drawer = ImageDrawer(self.image_shape)
        else:
            self.drawer = initial_drawer
        self.target_image = self.image_greyscale

    def set_sampling_mask(self, image_path):
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

    def __init__(self, n_strokes, brush_range, initial_fenotype = None):
        if initial_fenotype is None:
            self.n_strokes = n_strokes
            self.fenotype = []
        else:
            self.n_strikes = len(initial_fenotype)
            self.fenotype = initial_fenotype

        self.minSize = brush_range[0] #0.1 #0.3
        self.maxSize = brush_range[1] #0.3 # 0.7
        self.brushSide = 300 # brush image resolution in pixels
        self.padding = int(self.brushSide * self.maxSize / 2 + 5)

    def clone(self):
        return DrawingIndividual(self.n_strokes, brush_range=[self.minSize, self.maxSize], initial_fenotype=self.fenotype.copy())

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
            self.mutate_gene(drawing_problem, i)


    def mutate_gene(self, drawing_problem, index):
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
        self.hall_of_fame.update(population)
        fittest_image = self.drawer.draw(self.hall_of_fame[0].fenotype, self.hall_of_fame[0].padding)

        self.image_buffer.append(fittest_image)
    
        if self.show_progress_images is True:
            plt.clf()
            plt.imshow(fittest_image, cmap='gray')
            plt.show()

    def best_image(self):
        return self.image_buffer[-1]

class GreedyNotebookDrawingMonitor:

    def __init__(self, drawer, show_progress_images=True):
        self.show_progress_images = show_progress_images
        self.drawer = drawer
       
    def submit(self, population):
        hall_of_fame = HallOfFame(1) 
        hall_of_fame.update(population)

        self.fittest_image = self.drawer.draw(hall_of_fame[0].fenotype, hall_of_fame[0].padding)
    
        if self.show_progress_images is True:
            clear_output(wait=True)
            plt.imshow(self.fittest_image, cmap='gray')
            plt.show()

    def best_image(self):
        return self.fittest_image

class GeneticDrawing:

    def __init__(self, drawing_problem, n_parallel_jobs=-1):
        log.info("Initializing GeneticDrawing")

        self.drawing_problem = drawing_problem
        self.n_parallel_jobs=n_parallel_jobs

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
        self.toolbox.register("population", deap_init_population, list, self.toolbox.individual, drawing_problem = self.drawing_problem)

        # Custom mutation function
        self.toolbox.register("select", deap_select_population)

        # Custom mutation function
        self.toolbox.register("mutate", deap_mutate_individual)

        # Custom fitness function
        self.toolbox.register("evaluate", deap_evaluate_individual, drawing_problem=drawing_problem)


    def generate(self, n_generations=100, population_size=50, individual_size=10, monitor=None):
        log.info(f'Starting working. n_generations {n_generations}, population_size {population_size}, individual_size {individual_size}')

        try: 
            if population_size != 1:
                parallel_backend = parallel_backend('threading', n_jobs=self.n_parallel_jobs)

            population = self.toolbox.population(size=population_size, individual_size=individual_size)
            
            # Evaluate the entire population
            #fitnesses = map(toolbox.evaluate, population)
            fitnesses = Parallel()(delayed(self.toolbox.evaluate)(individual) for individual in population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            for g in range(n_generations):
                log.debug(f'Running generation {g}')

                # Select the next generation individuals
                offspring = self.toolbox.select(population, len(population))
                log.debug(f'g{g} offspring')

                # Clone the selected individuals
                offspring = [self.toolbox.clone(o) for o in offspring]
                #offspring = map(toolbox.clone, offspring)
                log.debug(f'g{g} clone')

                if (self.crossover_prob > 0):
                    # Apply crossover and mutation on the offspring
                    for child1, child2 in zip(offspring[::2], offspring[1::2]):
                        if random.random() < crossover_prob:
                            self.toolbox.mate(child1, child2)
                            del child1.fitness.values
                            del child2.fitness.values
                    log.debug(f'g{g} crossover')

                if (self.mutation_prob > 0):
                    for mutant in offspring:
                        if random.random() < self.mutation_prob:
                            self.toolbox.mutate(mutant, self.drawing_problem)
                            del mutant.fitness.values
                    log.debug(f'g{g} mutation')

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                #fitnesses = map(toolbox.evaluate, invalid_ind)
                fitnesses = Parallel()(delayed(self.toolbox.evaluate)(individual) for individual in invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # The population is entirely replaced by the offspring
                population[:] = offspring

                log.debug(f'{g} final population')
                if monitor is not None:
                    monitor.submit(population)


            return population
        finally:
            if parallel_backend is not None:
                parallel_backend.close()

class GreedyGeneticDrawing:
    
    def __init__(self, drawing_problem, brushes_range = DrawingBrushesRange(), n_parallel_jobs=-1):
        log.info("Initializing GreedyGeneticDrawing")

        self.drawing_problem = drawing_problem
        self.n_parallel_jobs = n_parallel_jobs
        self.brushes_range = brushes_range


    def generate(self, stages=100, n_generations=100, population_size=50, individual_size=10, drawer=None, show_progress_images=False):
        log.info(f'Greedy GA. {stages} stages, {n_generations} n_generations, {population_size} population_size, {individual_size} individual_size')

        if drawer is None:
            drawer = ImageDrawer(image_shape = self.drawing_problem.image_shape)

        for s in range(stages):
            monitor = GreedyNotebookDrawingMonitor(drawer, show_progress_images)

            brush_range = self.brushes_range.calculate_brush_range(s, stages) 
            log.info(f'Starting stage {s} with brush range {brush_range}')

            drawing_problem = DrawingProblem(color_image = self.drawing_problem.color_image, brush_range = brush_range, initial_drawer = drawer)

            ga = GeneticDrawing(drawing_problem, n_parallel_jobs = self.n_parallel_jobs)
            ga.generate(n_generations, population_size, individual_size, monitor)

            # Use best image as start of the iteration
            drawer = ImageDrawer(canvas = monitor.best_image())
            
            log.info(f'Stage {s} ended')

        log.info(f'Greedy GA ended')
        return monitor

class GreedyIterativeDrawing:
    
    def __init__(self, drawing_problem, brushes_range = DrawingBrushesRange(), n_parallel_jobs=-1):
        log.info("Initializing GreedyIterativeDrawing")

        self.drawing_problem = drawing_problem
        self.n_parallel_jobs = n_parallel_jobs
        self.brushes_range = brushes_range


    def generate(self, stages=100, n_trials=100, individual_size=10, drawer=None, show_progress_images=False):
        log.info(f'Greedy iterative search started. {stages} stages, {n_trials} n_trials, {individual_size} individual_size')

        if drawer is None:
            drawer = ImageDrawer(image_shape = self.drawing_problem.image_shape)


        for s in range(stages):
            brush_range = self.brushes_range.calculate_brush_range(s, stages) 

            drawing_problem = DrawingProblem(color_image = self.drawing_problem.color_image, brush_range = brush_range, initial_drawer = drawer)
            monitor = GreedyNotebookDrawingMonitor(drawer, show_progress_images)

            log.info(f'Starting stage {s} with brush range {brush_range}')

            individual = DrawingIndividual(n_strokes=individual_size, brush_range=brush_range)
            individual.init_random(drawing_problem)

            best_test_error = None
            best_image = None
            for t in range(n_trials):    
                for g in range(individual_size):
                    test = individual.clone()
                    test.mutate_gene(drawing_problem, g)

                    error, image = deap_evaluate_individual(test, drawing_problem)
                    if error < best_test_error:
                        individual = test
                        best_test_error = error
                        best_image = image

            # Restart search using the last previous best point
            drawer = ImageDrawer(canvas = best_image)
            log.info(f'Stage {s} ended')

        log.info(f'Greedy iterative search ended')
        return monitor
    

def deap_init_individual(icls, size, drawing_problem):
    individual = icls(n_strokes = size, brush_range = drawing_problem.brush_range)
    individual.init_random(drawing_problem)
    return individual

def deap_init_population(pcls, init_individual, size, individual_size, drawing_problem):
    return pcls(init_individual(individual_size, drawing_problem) for i in range(size))

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

    return diff, image
