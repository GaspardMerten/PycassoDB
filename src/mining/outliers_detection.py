import pandas as pd


def stopped_motors(source: pd.DataFrame):
    """
    Return a dataframe containing labels over the state of each motors

    Take Oil Pressure data and compare its value to the motors RPM:
    - If oil pressure is set to 0 and the linked motor is at 0 RPM, the motor is likely turned off
    - If oil pressure or RPM is set to 0 but the other is not, the motor is likely turned on, but faulty sensor
    """
    df = source.copy()

    # Is the motor stopped?
    df["motor_stopped_1"] = (df["oil_press_1"] == 0) & (df["rpm_1"] == 0)
    df["motor_stopped_2"] = (df["oil_press_2"] == 0) & (df["rpm_2"] == 0)

    # Is the RPM measurement faulty?
    df["faulty_rpm_1"] = (df["oil_press_1"] != 0) & (df["rpm_1"] == 0)
    df["faulty_rpm_2"] = (df["oil_press_2"] != 0) & (df["rpm_2"] == 0)

    # Is the Oil Pressure measurement faulty?
    df["faulty_oil_press_1"] = (df["oil_press_1"] == 0) & (df["rpm_1"] != 0)
    df["faulty_oil_press_2"] = (df["oil_press_2"] == 0) & (df["rpm_2"] != 0)

    # Are both motors stopped?
    df["train_stopped"] = df["motor_stopped_1"] & df["motor_stopped_2"]
    # Is the train stopped near a train station?
    df["stopped_at_station"] = (
        df["train_stopped"]
        & (df["classification"] == "Station")
        & (df["stop_distance"] <= 1000)
    )
    # Note: added a distance measure to ensure that we are not in a workshop (ensure closeness to train station)

    # Train stopped but not near a stop point of the train network
    df["stopped_at_workshop"] = df["train_stopped"] & (df["stop_distance"] > 1000)

    # Is only one of the motor stopped?
    df["eco_mode"] = df["motor_stopped_1"] ^ df["motor_stopped_2"]

    return df


def cooling_sensors(source: pd.DataFrame):
    df = source.copy()

    # Verify overheating of each sensor
    df["water_overheating_1"] = df["water_temp_1"] > 100
    df["water_overheating_2"] = df["water_temp_2"] > 100

    df["oil_overheating_1"] = df["oil_temp_1"] > 115
    df["oil_overheating_2"] = df["oil_temp_2"] > 115

    df["air_overheating_1"] = df["air_temp_1"] > 65
    df["air_overheating_2"] = df["air_temp_2"] > 65

    return df


def train_speed(source: pd.DataFrame):
    df = source.copy()
    # Verify speed of the train (120km/h is train max speed)
    df["speed_too_high"] = df["speed"] > 120

    return df


def train_stopped(source: pd.DataFrame, threshold: float):
    """
    Use the location of the nearest stop point to determine if the train is at rest
    """
    df = source.copy()

    threshold = 500  # distance in meters
    df["at_station"] = (df["classification"] == "Station") & (
        df["stop_distance"] < threshold
    )
    df["at_stop_point"] = (df["classification"] != "Station") & (df["stop_distance"] < 100)

    # Train under
    df["train_not_moving"] = (df["speed"] < 1) & (
        df["at_station"] | df["at_stop_point"]
    )

    return df
