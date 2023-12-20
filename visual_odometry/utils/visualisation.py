import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import params.params as params


def drawCamera(
    ax,
    position,
    direction,
    length_scale=1,
    head_size=10,
    equal_axis=True,
    set_ax_limits=True,
):
    # Draws a camera consisting of arrows into a 3d Plot
    # ax            axes object, creates as follows
    #                   fig = plt.figure()
    #                   ax = fig.add_subplot(projection='3d')
    # position      np.array(3,) containing the camera position
    # direction     np.array(3,3) where each column corresponds to the [x, y, z]
    #               axis direction
    # length_scale  length scale: the arrows are drawn with length
    #               length_scale * direction
    # head_size     controls the size of the head of the arrows
    # equal_axis    boolean, if set to True (default) the axis are set to an
    #               equal aspect ratio
    # set_ax_limits if set to false, the plot box is not touched by the function

    arrow_prop_dict = dict(mutation_scale=head_size, arrowstyle="-|>", color="r")
    a = Arrow3D(
        [position[0], position[0] + length_scale * direction[0, 0]],
        [position[1], position[1] + length_scale * direction[1, 0]],
        [position[2], position[2] + length_scale * direction[2, 0]],
        **arrow_prop_dict,
    )
    ax.add_artist(a)
    arrow_prop_dict = dict(mutation_scale=head_size, arrowstyle="-|>", color="g")
    a = Arrow3D(
        [position[0], position[0] + length_scale * direction[0, 1]],
        [position[1], position[1] + length_scale * direction[1, 1]],
        [position[2], position[2] + length_scale * direction[2, 1]],
        **arrow_prop_dict,
    )
    ax.add_artist(a)
    arrow_prop_dict = dict(mutation_scale=head_size, arrowstyle="-|>", color="b")
    a = Arrow3D(
        [position[0], position[0] + length_scale * direction[0, 2]],
        [position[1], position[1] + length_scale * direction[1, 2]],
        [position[2], position[2] + length_scale * direction[2, 2]],
        **arrow_prop_dict,
    )
    ax.add_artist(a)

    ax.text(
        position[0] + length_scale * direction[0, 0],
        position[1] + length_scale * direction[1, 0],
        position[2] + length_scale * direction[2, 0],
        "x",
        color="red",
    )

    ax.text(
        position[0] + length_scale * direction[0, 1],
        position[1] + length_scale * direction[1, 1],
        position[2] + length_scale * direction[2, 1],
        "y",
        color="green",
    )

    ax.text(
        position[0] + length_scale * direction[0, 2],
        position[1] + length_scale * direction[1, 2],
        position[2] + length_scale * direction[2, 2],
        "z",
        color="blue",
    )

    if not set_ax_limits:
        return

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    ax.set_xlim([min(xlim[0], position[0]), max(xlim[1], position[0])])
    ax.set_ylim([min(ylim[0], position[1]), max(ylim[1], position[1])])
    ax.set_zlim([min(zlim[0], position[2]), max(zlim[1], position[2])])

    # This sets the aspect ratio to 'equal'
    if equal_axis:
        ax.set_box_aspect(
            (np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim()))
        )


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def update_visualization(
    fig,
    ax1,
    ax2,
    ax3,
    current_image,
    state,
    previous_keypoint_locations,
    current_locations,
    global_point_cloud,
    R,
    t,
    camera_pose_history,
    num_added_landmarks,
    idx,
):
    """The main visualization function for the pipeline."""
    # Update Plot 1: Current image with keypoints
    ax1.clear()
    ax1.set_title(f"Image {idx}. With {num_added_landmarks} new landmarks")
    num_old_landmarks = state.P.shape[1] - num_added_landmarks

    idx = 0
    for p1, p2 in zip(previous_keypoint_locations, current_locations):
        color = (0, 255, 0)
        idx += 1
        # Convert to integer tuples
        p1 = (int(p1[0]), int(p1[1]))
        p2 = (int(p2[0]), int(p2[1]))

        cv2.circle(current_image, p1, 5, (0, 0, 255), -1)  # Blue for previous keypoints
        cv2.circle(current_image, p2, 5, color, -1)  # Green for new keypoints
        cv2.line(current_image, p1, p2, color, 2)  # line to indicate movement

    for idx, p in enumerate(state.P.T):
        color = (255, 132, 0) if idx < num_old_landmarks else (255, 0, 0)
        p = p.astype(int)
        cv2.circle(
            current_image,
            (p[0], p[1]),
            radius=5,
            color=color,
            thickness=-1,
        )

    if state.C is not None:
        for c in state.C.T:
            c = c.astype(int)
            cv2.circle(
                current_image,
                (c[0], c[1]),
                radius=2,
                color=(50, 168, 155),
                thickness=-1,
            )
    # Create proxy artists for the legend
    green_circle = Line2D(
        [0], [0], linestyle="none", marker="o", markersize=6, markerfacecolor=(0, 1, 0)
    )
    red_circle = Line2D(
        [0], [0], linestyle="none", marker="o", markersize=6, markerfacecolor=(1, 0, 0)
    )
    cyan_circle = Line2D(
        [0],
        [0],
        linestyle="none",
        marker="o",
        markersize=3,
        markerfacecolor=(50 / 255, 168 / 255, 155 / 255),
    )
    orange_cricle = Line2D(
        [0],
        [0],
        linestyle="none",
        marker="o",
        markersize=6,
        markerfacecolor=(237 / 255, 137 / 255, 7 / 255),  # Converted to range 0-1
    )

    # Add the legend to the plot
    ax1.legend(
        [orange_cricle, green_circle, red_circle, cyan_circle],
        [
            "Landmarks tracked in front of cam.",
            "Landmarks tracked behind cam, removed",
            "Promoted Candidates",
            "Candidates",
        ],
        numpoints=1,
        fontsize="small",
    )
    # equalized_image = cv2.equalizeHist(current_image)
    ax1.imshow(current_image)

    # Update Plot 2: 3D Point Cloud with Camera Pose
    ax2.clear()
    ax2.set_title("3D Point Cloud with Camera Pose")
    ax2.set_xlabel("X axis")
    ax2.set_ylabel("Y axis")
    ax2.set_zlabel("Z axis")

    if params.GLOBAL_POINT_CLOUD:
        global_x = [point[0] for point in global_point_cloud]
        global_y = [point[1] for point in global_point_cloud]
        global_z = [point[2] for point in global_point_cloud]
        ax2.scatter(global_x, global_y, global_z, s=2)
        ax2.set_box_aspect((np.ptp(global_x), np.ptp(global_y), np.ptp(global_z)))
    else:
        ax2.scatter(state.X[0, :], state.X[1, :], state.X[2, :])
        ax2.set_box_aspect(
            (np.ptp(state.X[0, :]), np.ptp(state.X[1, :]), np.ptp(state.X[2, :]))
        )
    drawCamera(
        ax2,
        (-R.T @ t).ravel(),
        R.T,
        length_scale=20,
        head_size=10,
        equal_axis=False,
        set_ax_limits=False,
    )

    # Add a marker for the origin
    ax2.scatter([0], [0], [0], color="k", marker="o")  # Black dot at the origin

    ax2.plot(
        camera_pose_history[0, :],
        camera_pose_history[1, :],
        camera_pose_history[2, :],
        color="red",
    )  # Line plot

    # Update Plot 3: 2D Camera Pose History
    ax3.clear()
    ax3.set_title("2D Top-Down Camera Pose History")
    ax3.set_xlabel("X axis")
    ax3.set_ylabel("Z axis")
    ax3.plot(camera_pose_history[0, :], camera_pose_history[2, :])
    ax3.scatter(
        camera_pose_history[0, -1],
        camera_pose_history[2, -1],
        color="g",
        marker="o",
        s=4,
    )  # Marker for current camera location
    ax3.set_aspect("equal")
    ax3.autoscale(enable=True, axis="both")

    plt.draw()
    plt.pause(0.0001)  # Necessary for the plot to update
    if params.WAIT_ARROW:
        fig.canvas.start_event_loop(timeout=-1)
    else:
        # do nothing since we are going to compute the next frame
        pass
