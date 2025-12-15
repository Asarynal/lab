from sklearn.datasets import make_blobs
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pandas as pd

def random_point_in_circle(cx, cy, r):
    """
    One random point inside a circle of radius r centered at (cx, cy).
    """
    theta = random.uniform(0, 2 * math.pi)
    u = random.random()          # in [0,1)
    radius = r * math.sqrt(u)    # sqrt for uniform area

    x = cx + radius * math.cos(theta)
    y = cy + radius * math.sin(theta)
    return (x, y)


def random_points_in_circle(cx, cy, r, density_factor):
    n_points = round(density_factor*r*r)
    return [random_point_in_circle(cx, cy, r) for _ in range(n_points)]


def generate_blobs_csv(
    n_samples=200,
    n_clusters=5,
    cluster_std=0.30,
    random_state=42,
    filename="blobs.csv"
):
    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=random_state
    )

    # Save as CSV: x, y, cluster_id
    data = np.column_stack([X, y])
    np.savetxt(filename, data, delimiter=",", header="x,y,cluster_id", comments="")
    print(f"Saved dataset to {filename}")
    return X, y

# Example run
if __name__ == "__main__":
    # generate_blobs_csv()
    density_factor = 100
    circle1 = [[p[0],p[1],1] for p in random_points_in_circle(2, 4, 1, density_factor)]
    circle2 = [[p[0],p[1],2] for p in random_points_in_circle(5, 9, 4, density_factor)]
    circle3 = [[p[0],p[1],3] for p in random_points_in_circle(5, 2, 2, density_factor)]
    circle4 = [[p[0],p[1],4] for p in random_points_in_circle(9, 4, 2.1, density_factor)]
    circle5 = [[p[0],p[1],5] for p in random_points_in_circle(8, 1, 0.9, density_factor)]
    circle6 = [[p[0],p[1],6] for p in random_points_in_circle(1.7, 1.5, 0.9, density_factor)]
    blobs = [*circle1,*circle2,*circle3,*circle4,*circle5,*circle6]
    xpoints = [p[0] for p in blobs]
    ypoints = [p[1] for p in blobs]
    df = pd.DataFrame(blobs,columns=["x","y","cluster_id"])
    df.to_csv("blobs.csv", index=False)
    print("Saved blobs.csv")
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.scatter(xpoints,ypoints,s=8,alpha=0.85)
    # plt.grid(True, linestyle=':', linewidth=0.6)
    # plt.show()

    
