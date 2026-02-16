import math
import numpy as np
import matplotlib.pyplot as plt


class Mapper:

    def __init__(self, pipe_seg_length, standard_turn_angles):
        self.pipe_seg_length = pipe_seg_length
        self.standard_turn_angles = standard_turn_angles

        initial_junction0 = {
            'angle': 0.0,  # Initial angle is 0 degrees, the angle is the angle that preceed the straight segment
            'pos': np.array([0.0, 0.0]),  # Initial position at the origin
            'id': 0  # Initial junction ID
            }
        initial_junction1 = {
            'angle': 0.0,  # Initial angle is 0 degrees, the angle is the angle that preceed the straight segment
            'pos': np.array([0.0, self.pipe_seg_length]),  # Initial position at the origin
            'id': 1  # Initial junction ID
            }
        
        self.junctions = [initial_junction0, initial_junction1]

        # Points used to plot the map
        self.map_points = [
            {'pos': initial_junction0['pos'], 'type': 'junction'},
            {'pos': initial_junction1['pos'], 'type': 'junction'}
        ]

        # Turns geometry look-up table
        self.turns_geometry = {
            'straight': {'angle' : 0.0, 'first_seg' : 0.0, 'second_seg' : 0.0},
            'left45': {'angle' : -45.0, 'first_seg' : 7.5, 'second_seg' : 2.5},
            'left90': {'angle' : -90.0, 'first_seg' : 9.0, 'second_seg' : 5.0},
            'right45': {'angle' : 45.0, 'first_seg' : 7.5, 'second_seg' : 2.5},
            'right90': {'angle' : 90.0, 'first_seg' : 9.0, 'second_seg' : 5.0},
        }

    def add_junction(self, direction, junction_id):
        if junction_id in [j['id'] for j in self.junctions]:
            return  # Skip if already added

        # Get last junction and current heading
        last_junction = self.junctions[-1]
        current_heading = np.sum([j['angle'] for j in self.junctions])
        current_heading_rad = np.deg2rad(current_heading)

        # Lookup turn geometry
        if direction not in self.turns_geometry:
            raise ValueError(f"Unknown direction '{direction}'")
        turn_info = self.turns_geometry[direction]
        angle = turn_info['angle']
        first_seg = turn_info['first_seg']
        second_seg = turn_info['second_seg']

        # 1. Go forward by first turn segment (in current heading)
        pre_turn_offset = np.array([
            -first_seg * np.cos(math.pi / 2 + current_heading_rad),
            first_seg * np.sin(math.pi / 2 + current_heading_rad)
        ])
        pre_turn_pos = last_junction['pos'] + pre_turn_offset
        self.map_points.append({'pos': pre_turn_pos, 'type': 'pre_turn'})

        # 2. Apply turn
        new_heading = current_heading + angle
        new_heading_rad = np.deg2rad(new_heading)

        # 3. Go forward by second turn segment (in new heading)
        post_turn_offset = np.array([
            -second_seg * np.cos(math.pi / 2 + new_heading_rad),
            second_seg * np.sin(math.pi / 2 + new_heading_rad)
        ])
        post_turn_pos = pre_turn_pos + post_turn_offset
        self.map_points.append({'pos': post_turn_pos, 'type': 'post_turn'})

        # 4. Go forward by main pipe segment (in new heading)
        junction_offset = np.array([
            -self.pipe_seg_length * np.cos(math.pi / 2 + new_heading_rad),
            self.pipe_seg_length * np.sin(math.pi / 2 + new_heading_rad)
        ])
        new_junction_pos = post_turn_pos + junction_offset

        print(f'new junction pos: {new_junction_pos}')

        # Add to map_points and junctions
        self.map_points.append({'pos': new_junction_pos, 'type': 'junction'})
        self.junctions.append({
            'angle': angle,
            'pos': new_junction_pos,
            'id': junction_id
        })

    def round_turn_angle(self, angle):
        """
        Round the turn angle to the closest standard turn angle.
        """
        if not self.standard_turn_angles:
            return angle
        # Find the closest standard turn angle
        closest_angle = min(self.standard_turn_angles, key=lambda x: abs(x - angle))
        # Check if the angle is within 10 degrees of the closest standard angle
        print(f"Rounding angle {angle} to closest standard angle {closest_angle}")
        return closest_angle

    def initialize_map_plot(self):
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Map")
        return fig, ax
    
    def update_map_plot(self, ax, trajectory, orientation=None):
        ax.cla()

        ax.set_xlabel("X (sideways)")
        ax.set_ylabel("Z (forward)")
        ax.set_title("Trajectory")

        # ax.autoscale_view()
        ax.set_xlim(-50, 50)
        ax.set_ylim(-10, 90)

        # Plot trajectory
        if len(trajectory) > 2:

            # Extract X and Z coordinates from poses
            xs = [pose[0] for pose in trajectory]
            zs = [pose[2] for pose in trajectory]

            ax.cla()
            ax.scatter(xs, zs, color='#728880', s=8, label="Estimated position")
            # ax.scatter(xs[-1], zs[-1], color='red', label="Current Pose")

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.legend()

            ax.set_xlim(-50, 50)
            ax.set_ylim(-10, 90)

            if orientation is not None and orientation.shape == (3, 3):
                origin = np.array([xs[-1], zs[-1]])

                # Get the current axis span to scale arrow size
                x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
                z_span = ax.get_ylim()[1] - ax.get_ylim()[0]
                arrow_scale = 0.1 * max(x_span, z_span)  # 10% of larger axis span

                # Extract forward direction (Z-axis of rotation matrix)
                forward_vector = orientation[:, 2]
                forward_2d = np.array([forward_vector[0], forward_vector[2]])

                # Normalize and scale
                norm = np.linalg.norm(forward_2d)
                if norm > 1e-6:
                    forward_2d = (forward_2d / norm) * arrow_scale
                    ax.arrow(
                        origin[0], origin[1],
                        forward_2d[0], forward_2d[1],
                        head_width=0.05 * arrow_scale,
                        head_length=0.1 * arrow_scale,
                        fc='green', ec='green', label='Orientation'
                    )

        # Plot map points
        if len(self.map_points) > 1:
            # Plot line through all points
            map_positions = np.array([pt['pos'] for pt in self.map_points])
            ax.plot(map_positions[:, 0], map_positions[:, 1], color='orange', linestyle='--', linewidth=2)

            # Optionally color-code by point type
            for pt in self.map_points:
                x, y = pt['pos']
                pt_type = pt.get('type', 'unknown')

                if pt_type == 'junction':
                    ax.scatter(x, y, color='red', s=30, marker='x')
                if pt_type == 'pre_turn' or pt_type == 'post_turn':
                    ax.scatter(x, y, color='green', s=30, marker='o')

        # artificial junction and network line for label
        ax.scatter(0.0, 0.0, marker='x', color = 'red', label='Junction')
        ax.plot(0, 25, color='orange', linestyle='--', linewidth=2, label='Pipe Network')

        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(0.1)


