import geopy
import math
from tqdm import trange
import pandas as pd
print(geopy.distance.distance((lat_start, lon_ziel), (lat_ziel, lon_ziel)).meters)




"""
Berechnung mit bisheriger Methode
"""
def compute_interpolation(lat_start, lon_start, lat_ziel, lon_ziel, D, d):
	lat_int = lat_start + ((d/D) * (lat_ziel-lat_start))
	lon_int = lon_start + ((d/D) * (lon_ziel-lon_start))
	return lat_int, lon_int

"""
Berechnung mit anderer Methode
"""
def compute_wsg(lat_start, lon_start, lat_ziel, lon_ziel, D, d):
	bearing = calculate_initial_compass_bearing((lat_start, lon_start),(lat_ziel, lon_ziel))
	distm = (distance.distance((lat_start, lon_start), (lat_ziel, lon_ziel)).meters/D)*d # (6088.745955951135/782.84)*480
	geo_lat, geo_lon, _ = geopy.distance.distance(meters=distm).destination((lat_start, lon_start), bearing=bearing)
	return geo_lat, geo_lon


# start koord
# latitude \in [-90, 90]
# longitude \in [-180, 180]
# Abweichung : -0.0005 bis 0.0005
result = []
for i in trange(10):
	lat_epsilon = random.uniform(-0.005,0.005)
	lon_epsilon = random.uniform(-0.005,0.005)
	lat_start = random.uniform(-90,90)
	lon_start = random.uniform(-180,180)
	lat_ziel = lat_start + (lat_start*lat_epsilon)
	lon_ziel = lon_start + (lon_start*lon_epsilon)
	D = 1000
	d = random.uniform(1,1000)
	start_inter = time.time()
	lat_int, lon_int = compute_interpolation(lat_start, lon_start , lat_ziel, lon_ziel, D, d)
	end_inter = time.time()
	geo_lat, geo_lon = compute_wsg(lat_start, lon_start , lat_ziel, lon_ziel, D, d)
	end_wsg = time.time()
	result.append({"lat_start":lat_start, "lon_start":lon_start, "lat_ziel":lat_ziel, "lon_ziel":lon_ziel, "lat_epsilon":lat_epsilon, "lon_epsilon": lon_epsilon, 
		"distance":geopy.distance.distance((lat_start, lon_start), (lat_ziel, lon_ziel)).meters, "error":geopy.distance.distance((lat_int, lon_int), (geo_lat, geo_lon)).meters,
		"time_inter":f"{end_inter-start_inter:.7f}", "time_wsg":f"{end_wsg-end_inter:.7f}"
		})


df = pd.DataFrame(result)
df



lat_start, lon_start = 2.49651, 21.61419
lat_ziel, lon_ziel = 2.49651, 21.88419
D = 900
d = 400
lat_int, lon_int = compute_interpolation(lat_start, lon_start , lat_ziel, lon_ziel, D, d)
geo_lat, geo_lon = compute_wsg(lat_start, lon_start , lat_ziel, lon_ziel, D, d)

print(f"Distance: {geopy.distance.distance((lat_start, lon_start), (lat_ziel, lon_ziel)).meters} , Error: {geopy.distance.distance((lat_int, lon_int), (geo_lat, geo_lon)).meters}")



(55.51197069129917, 1.6686182240762812)
(55.512011734283604, 1.6685693672922886)


def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
    θ = atan2(sin(Δlong).cos(lat2),
              cos(lat1).sin(lat2) − 
    sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
    - `pointA: The tuple representing the 
    latitude/longitude for the
    first point. Latitude and longitude must be in 
    decimal degrees
    - `pointB: The tuple representing the latitude/longitude for the
    second point. Latitude and longitude must be in decimal degrees
    :Returns:
    The bearing in degrees
    :Returns Type:
    float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")
    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])
    diffLong = math.radians(pointB[1] - pointA[1])
    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
        * math.cos(lat2) * math.cos(diffLong))
    initial_bearing = math.atan2(x, y)
    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing

