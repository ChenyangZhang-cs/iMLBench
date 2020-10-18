import matplotlib.pyplot as plt
import subprocess
import sys

class Result:
  rsquared = 0
  time = 0
  a0 = 0
  a1 = 0

  def __init__(self, parsed):
    self.a0 = float(parsed[0])
    self.a1 = float(parsed[1])
    self.time = float(parsed[2])
    self.rsquared = int(parsed[3])

  def print(self):
    print('\t| A0: ' + str(self.a0) + '\n\t| A1: ' + str(self.a1) + '\n\t| Time: ' + str(self.time) + ' ms\n\t| R-Squared: ' + str(self.rsquared) + '%\n\t| Equation: ' + str(self.a1) + 'x + ' + str(self.a0) + '\n')

class Dataset:
  x_values = []
  y_values = []
  parallele_result = None
  iterative_result = None
  x_limits = None

  def __init__(self, filename, x_limits):
    self.x_limits = x_limits
    file = open('assets/' + filename + '.txt', 'r')
    self.read_file(file)
    file.close()
    plt.scatter(self.x_values, self.y_values)
    self.read_results()
    self.plot_function(self.parallele_result)

  def print(self):
    print('\tParallelisation\n\t-----------------')
    self.parallele_result.print()
    print('\tIterative\n\t-----------------')
    self.iterative_result.print()

  def read_file(self, file):
    while True:
      line = file.readline()
      if not line:
        break
      str_x, str_y = line.split('\t')
      self.x_values.append(float(str_x))
      self.y_values.append(float(str_y))

  def read_results(self):
    while True:
      try:
        file = open('assets/_results.txt', 'r')
        results = []
        for i in range(0, 4):
          results.append(Result(file.readline().split('#')))
        file.close()
        return results
      except:
        subprocess.run(["./build/linear", "-no_print"])

  def plot_function(self, result):
    plt.plot(self.x_limits, [result.a0, result.a0 + self.x_limits[1] * result.a1], linestyle='--', c='#000000')

  def show(self):
    plt.show()

class House_Dataset (Dataset):
  def __init__(self):
    super().__init__('house', [0, 250])
    plt.xlabel('Surface')
    plt.ylabel('Loyer')

  def read_results(self):
    results = super().read_results()
    self.parallele_result = results[0]
    self.iterative_result = results[1]

  def print(self):
    print("\n> HOUSE REGRESSION (545):\n")
    super().print()

class Temperature_Dataset (Dataset):
  def __init__(self):
    super().__init__('temperature', [-20, 40])
    plt.xlabel('Température')
    plt.ylabel('Pression Atmosphérique')

  def read_results(self):
    results = super().read_results()
    self.parallele_result = results[2]
    self.iterative_result = results[3]

  def print(self):
    print("\n> TEMPERATURE REGRESSION (96453)\n")
    super().print()

if __name__ == '__main__':
  arg = sys.argv[1]
  dataset = None

  if arg == 'home':
    dataset = House_Dataset()
  elif arg == 'temperature':
    dataset = Temperature_Dataset()
  else:
    print('Error no dataset find for ' + arg)
    exit(1)

  dataset.print()
  dataset.show()
