import requests


class Chromosome:
    URL_BASE = "https://aydanomachado.com/mlclass/02_Optimization.php?"
    KEY = "&dev_key=Café%20com%20leite"

    def __init__(self, angles=None, score=0, uncovered=False):
        self.uncovered = uncovered
        self.score = score
        if angles is None:
            angles = []
        self.angles = angles

    def __repr__(self):
        return str(self.score) + " P1 = " + str(self.angles[0]) + " T1 = " + str(self.angles[1]) \
               + " P2 = " + str(self.angles[2]) + " T2 = " + str(self.angles[3]) \
               + " P3 = " + str(self.angles[4]) + " T3 = " + str(self.angles[5])

    def comparator(self, other):

        if not self.uncovered:
            url = self.URL_BASE + "phi1=" + str(self.angles[0]) + "&theta1=" + str(self.angles[1]) \
                  + "&phi2=" + str(self.angles[2]) + "&theta2=" + str(self.angles[3]) \
                  + "&phi3=" + str(self.angles[4]) + "&theta3=" + str(self.angles[5]) + self.KEY
            r = requests.get(url=url)
            data = r.json()
            self.score = float(data['gain'])
            self.uncovered = True

        if not other.uncovered:
            url = self.URL_BASE + "phi1=" + str(other.angles[0]) + "&theta1=" + str(other.angles[1]) \
                  + "&phi2=" + str(other.angles[2]) + "&theta2=" + str(other.angles[3]) \
                  + "&phi3=" + str(other.angles[4]) + "&theta3=" + str(other.angles[5]) + self.KEY
            r = requests.get(url=url)
            data = r.json()
            other.score = float(data['gain'])
            other.uncovered = True

        if self.score > other.score:
            return -1
        return 1
