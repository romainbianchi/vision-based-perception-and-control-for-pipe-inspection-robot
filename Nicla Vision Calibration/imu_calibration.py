import numpy as np

def load_data(filename='imu_calib_data.txt'):
    """
    Loads and parses IMU data from a file where each line is:
    [accel_x, accel_y, accel_z], [gyro_x, gyro_y, gyro_z]
    """
    accel_data = []
    gyro_data = []
    line_count = 0
    skipped_lines = 0

    with open(filename, 'r') as file:
        for line in file:
            line_count += 1
            try:
                parts = line.strip().split("], [")
                accel_str = parts[0].strip(' [')
                gyro_str = parts[1].strip(' ]\n')

                accel = [float(value) for value in accel_str.split()]
                gyro = [float(value) for value in gyro_str.split()]

                accel_data.append(accel)
                gyro_data.append(gyro)

            except Exception as e:
                skipped_lines += 1
                print(f"Skipping line {line_count}: {e}")

    print(f"\nSuccessfully parsed {line_count - skipped_lines}/{line_count} lines")
    return np.array(accel_data), np.array(gyro_data)


def calculate_offsets(accel_data, gyro_data):
    accel_data[:, 0] -= 9.81
    return {
        'accel_offset': np.mean(accel_data, axis=0),
        'gyro_offset': np.mean(gyro_data, axis=0),
        'accel_std': np.std(accel_data, axis=0),
        'gyro_std': np.std(gyro_data, axis=0)
    }


def save_calibration(offsets, filename='imu_calibration_results.txt'):
    with open(filename, 'w') as file:
        file.write("IMU Calibration Results\n")
        file.write("========================\n\n")
        file.write(f"Accelerometer Offset (m/s²): {offsets['accel_offset']}\n")
        file.write(f"Accelerometer Std Dev: {offsets['accel_std']}\n\n")
        file.write(f"Gyroscope Offset (dps): {offsets['gyro_offset']}\n")
        file.write(f"Gyroscope Std Dev: {offsets['gyro_std']}\n")


def main():
    try:
        accel_data, gyro_data = load_data('imu_calib_data.txt')
        print(f"Loaded {len(accel_data)} IMU samples")

        offsets = calculate_offsets(accel_data, gyro_data)

        print("\nCalibration Results:")
        print(f"Accelerometer Offset (m/s²): {offsets['accel_offset']}")
        print(f"Accelerometer Std Dev: {offsets['accel_std']}")
        print(f"Gyroscope Offset (dps): {offsets['gyro_offset']}")
        print(f"Gyroscope Std Dev: {offsets['gyro_std']}")

        save_calibration(offsets)
        print("\nCalibration results saved to 'imu_calibration_results.txt'")

    except FileNotFoundError:
        print("Error: Could not find 'imu_calib_data.txt'")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
