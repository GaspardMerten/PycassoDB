import pandas as pd

from src.framework import Component
from src.mining.outliers_detection import stopped_motors


class OilPressureNotMatchingRPMsComponent(Component):
    def run(self, source: pd.DataFrame):
        df = stopped_motors(source)

        df = df[df['faulty_oil_press_1'] | df['faulty_oil_press_2'] | df['faulty_rpm_1'] | df['faulty_rpm_2']]

        return df
