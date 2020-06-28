# Genetic Drawing
This is a refactor of https://github.com/anopara/genetic-drawing

@anopara inspired me to investigate about this subject.
This refactor includes:
- New Genetic Algorithm structures.
- Support to tune operators.
- Two notebooks to test the original algorithm against the new implementation.
- Conda environment.
- Logging and other tools to ease development.

You can follow up this topic at https://sergiomoratilla.com/2020-06-28-teaching-a-computer-to-draw/

## Python
There is a Conda environment included with this project.
Just do
```
conda env create -n genetic-drawing -f conda-environment.yml
conda activate genetic-drawing
```

To start, open either of the notebooks:
- RandomGreedyDrawing: original algorithm reimplemented.
- GeneticDrawing.ipynb: new genetic algorithm implementation.

## Credits
- Base GA code: https://github.com/anopara/genetic-drawing
- StopWatch: https://github.com/ravener/stopwatch.py
- Thanks to https://github.com/danieloop for listening my Python complains
