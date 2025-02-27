#!/usr/bin/env python3

from skyfield.api import Topos, load, EarthSatellite
import maidenhead as mh
from datetime import datetime, timedelta
import requests
import matplotlib.pyplot as plt
import numpy as np

# spacial reference cheat sheet
# Geocentric: used to describe a satellite from the earths center in x,y,z format
# Topocentric: used to describe the satellite position in Az, El, and moving towards/away at X m/s

# Constants
C = 299792458  # speed of light in m/s
NUM_POINTS = 50  # number of points to calculate along a satellite path
EVENT_AOS = 0
EVENT_TCA = 1
EVENT_LOS = 2
HTTP_STATUS_OK = 200

class Sat:
    def __init__(self, satID):
        self.satID = satID # NORAD ID
        self.satName = "None"
        self.tleLine1 = ""
        self.tleLine2 = ""

        # events are components of a pass
        self.events = []
        self.eventTimes = []

        # timescale
        self.ts = None

        # satellite’s orbit in a geocentric frame (relative to Earth’s center)
        self.satellite = None

        # topocentric reference point at the user’s location
        self.observer = None

    # get TLE from Celestrack.
    def getTLE(self):
        url = "https://celestrak.org/NORAD/elements/gp.php?CATNR=" + self.satID + "&FORMAT=TLE"
        
        # request the TLE from Celestrak
        # data returned is three lines with the first line being the satellite name
        response = requests.get(url)

        # make sure we have valid data
        # Celestrak returns "No GP data found" if you have an invalid NORAD ID
        if response.status_code != HTTP_STATUS_OK or "No GP data found" in response.text:
            raise Exception(f"Failed to fetch TLE data for {self.satID}")
        
        # strip out whitepace
        lines = response.text.strip().split('\n')

        # if we got less than three lines, something went wrong
        if len(lines) < 3:
            raise Exception("Invalid TLE data received")
        
        # store the data
        self.satName = lines[0].strip()
        self.tleLine1 = lines[1].strip()
        self.tleLine2 = lines[2].strip()

    # calculate future passes
    def calculate_passes(self, lat, lon, num_passes=10, min_degrees=10.0):
        self.ts = load.timescale()

        # defines the satellite’s orbit in a geocentric frame (relative to Earth’s center)
        self.satellite = EarthSatellite(self.tleLine1, self.tleLine2, self.satName, self.ts)

        # create a topocentric reference point at the user’s location
        self.observer = Topos(latitude_degrees=lat, longitude_degrees=lon)
        
        # computes the vector difference between the satellite’s geocentric position
        # and the observer’s topocentric position.
        # this gives a vector pointing from the observer to the satellite
        self.difference = self.satellite - self.observer
 
        # start from current time
        t0 = self.ts.now()

        # look ahead 7 days
        t1 = self.ts.utc(t0.utc_datetime() + timedelta(days=7))
        
        eventTimes, events = self.satellite.find_events(self.observer, t0, t1, altitude_degrees=min_degrees)
               
        passes = []
        current_pass = SatPass(self)
 
        for time, event in zip(eventTimes, events):
            # Calculate position at this time
            topocentric = self.difference.at(time)
            el, az, _ = topocentric.altaz()
            
            # events are in order AOS, TCA, LOS
            # when you reach LOS, the pass is complete
            if event == EVENT_AOS:
                current_pass.aos_time = time
                current_pass.aos_az = az.degrees
                current_pass.aos_el = el.degrees
            elif event == EVENT_TCA: 
                current_pass.tca_time = time
                current_pass.tca_az = az.degrees
                current_pass.tca_el = el.degrees
            elif event == EVENT_LOS:
                current_pass.los_time = time
                current_pass.los_az = az.degrees
                current_pass.los_el = el.degrees

                # add the pass to the collection
                passes.append(current_pass)

                # prep for the next pass as long as we haven't hit the limit
                if len(passes) < num_passes:
                    current_pass = SatPass(self)
            
            # check to see if we have calculated requested number of passes
            if len(passes) >= num_passes:
                break

        return passes
        
class SatPass:
    def __init__(self, sat):
        self.sat = sat

        self.azimuths, self.elevations = [], []
        self.doppler_up, self.doppler_down = [], []

        self.aos_time = None
        self.aos_az = None
        self.aos_el = None

        self.tca_time = None
        self.tca_az = None
        self.tca_el = None

        self.los_time = None
        self.los_az = None
        self.los_el = None

    def calculate_doppler(self, uplink_freq, downlink_freq):
        # calculate the duration of the satellite pass
        duration = (self.los_time.utc_datetime() - self.aos_time.utc_datetime()).total_seconds()

        # calculate the time fop num_points amount of steps throughout the pass
        time_steps = [self.sat.ts.utc(self.aos_time.utc_datetime() + 
                                      timedelta(seconds=i * duration / NUM_POINTS)) 
                                      for i in range(NUM_POINTS + 1)]
      
        self.doppler_up, self.doppler_down = [], []        

        for t in time_steps:
            # evaluates this difference at a specific time, result is in topocentric format
            topocentric = self.sat.difference.at(t)

            # position vector of the satellite relative to the observer
            # expressed in meters, within the topocentric coordinate system
            position = topocentric.position.m

            # gives the satellite’s velocity vector in the topocentric frame
            # This velocity is relative to the observer, not the Earth’s center.
            velocity = topocentric.velocity.m_per_s

            # calculate the radial velocity
            # only the component of the relative velocity that is directly towards
            # or away from the observer is considered in the calculation
            rv = np.dot(velocity, position) / np.linalg.norm(position)
            
            # calculate the doppler shift
            dopplerUp = uplink_freq * rv / C
            dopplerDown = -downlink_freq * rv / C

            # add to the collection
            self.doppler_up.append(dopplerUp)
            self.doppler_down.append(dopplerDown)

    def calculate_path(self):
        # calculate the duration of the satellite pass
        duration = (self.los_time.utc_datetime() - self.aos_time.utc_datetime()).total_seconds()

        # calculate the time fop num_points amount of steps throughout the pass
        time_steps = [self.sat.ts.utc(self.aos_time.utc_datetime() + 
                                      timedelta(seconds=i * duration / NUM_POINTS)) 
                                      for i in range(NUM_POINTS + 1)]
        
        self.azimuths, self.elevations = [], []
 
        for t in time_steps:
            # evaluates this difference at a specific time, result is in topocentric format
            topocentric = self.sat.difference.at(t)

            # converts the topocentric position into az and el
            el, az, _ = topocentric.altaz()

            # add to the collection
            self.azimuths.append(az.degrees)
            self.elevations.append(el.degrees)

    def plot_polar_flight_path(self, title):
        # create the plot, 11 inches wide, 8 inches tall
        fig = plt.figure(figsize=(11, 8))

        # add the subplot starting at 1,1,1 with a polar projection
        # polar projection is used to show the path of the satellite from
        # the viewpoint of the oberserver looking straight up. The center 
        # of the plot is directly overhead and the horizon is at the edges
        ax = fig.add_subplot(111, projection='polar')
        
        # set polar plot orientation
        ax.set_theta_zero_location('N')

        # set to clockwise, it defaults to counter-clockwise
        ax.set_theta_direction(-1)
        
        # plot path, "b-" is for a solid blue line
        az_rad = np.radians(self.azimuths)
        ax.plot(az_rad, [90 - ii  for ii in self.elevations], 'b-', label=f'{self.sat.satName} Path')
        
        # plot key points
        # go is green circle, ro is red circle, yo is yellow circle
        ax.plot(np.radians(self.aos_az), 90 - self.aos_el, 'go', label=f'AOS (Az: {self.aos_az:.1f}°)')
        ax.plot(np.radians(self.tca_az), 90 - self.tca_el, 'ro', label=f'TCA (Az: {self.tca_az:.1f}°, El: {self.tca_el:.1f}°)')
        ax.plot(np.radians(self.los_az), 90 - self.los_el, 'yo', label=f'LOS (Az: {self.los_az:.1f}°)')
        
        # set view limit
        ax.set_ylim(0, 90)

        # add circles every 30 degrees of elevation
        ax.set_yticks(range(0, 91, 30))

        # need 90-y to put 90 in the center, 0 on the edge
        ax.set_yticklabels([f'{90-y}°' for y in range(0, 91, 30)])
        
        # add the title
        ax.set_title(f"{title}\n"
                    f"AOS: {self.aos_time.utc_datetime().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
                    f"LOS: {self.los_time.utc_datetime().strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n")
        
        # place the legend top left, just outside the subplot
        ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.1))
        
        # automatically adjust subplot params so that the it fits in to the figure area
        plt.tight_layout()

        # save the plot
        # TODO: return the plot and let the calling function save it
        plt.savefig("plot.png")

# validate and format maidenhead grid
# this is a little overkill for this program
def format_maidenhead(grid):
    # remove whitespace
    grid = grid.strip()
    
    # make sure there are 6 chars
    if len(grid) != 6:
        raise ValueError("Maidenhead grid square must be 6 characters long")
    
    # extract parts and put in propper case
    field = grid[0:2].upper()
    square = grid[2:4]
    subsquare = grid[4:6].lower()
    
    # validate format is AA##aa
    if not (field.isalpha() and square.isdigit() and subsquare.isalpha()):
        raise ValueError("Invalid Maidenhead format: must be 2 letters, 2 numbers, 2 letters")
    
    return field + square + subsquare

# convert from maidenhead to lat, long
def maidenhead_to_latlon(grid):
    lat, lon = mh.to_location(grid, center=True)
    return lat, lon

def main():
    # ask for NORAD ID
    sat_id = input("Enter NORAD ID Number: ").strip()

    # get the frequencies in Mhz and convert to Hz
    uplink_freq = float(input("Enter uplink center frequency (MHz): ")) * 1e6
    downlink_freq = float(input("Enter downlink center frequency (MHz): ")) * 1e6

    # ask for 6-digit maidenhead grid
    grid = input("Enter your 6-digit Maidenhead grid square: ").strip()
    grid = format_maidenhead(grid)

    # convert to lat/lon
    lat, lon = maidenhead_to_latlon(grid)
    print(f"Location: {lat:.4f}°N, {lon:.4f}°E")
    
    sat = Sat(satID=sat_id)

    # fetch TLE data
    print("Fetching latest TLE data from CelesTrak...")
    sat.getTLE()
    print(f"{sat.satName}")
    print(f"TLE Line 1: {sat.tleLine1}")
    print(f"TLE Line 2: {sat.tleLine2}")

    # calculate passes
    print("\nCalculating next 10 passes...")
    passes = sat.calculate_passes(lat, lon, num_passes=10, min_degrees=1.0)
    
    # show a table header
    print("-" * 101)
    print(f"{'#':>2} {'AOS Time (UTC)':<20} {'AOS Az':>7}   {'TCA Time (UTC)':<20} {'TCA El':>7} {'TCA Az':>7}   {'LOS Time (UTC)':<20} {'LOS Az':>7}")
    print("-" * 101)
    
    # loop through passes
    for i, p in enumerate(passes, 1):
        aos_time = p.aos_time.utc_datetime().strftime('%Y-%m-%d %H:%M:%S')
        tca_time = p.tca_time.utc_datetime().strftime('%Y-%m-%d %H:%M:%S')
        los_time = p.los_time.utc_datetime().strftime('%Y-%m-%d %H:%M:%S')
        
        #print pass information
        print(f"{i:2d} {aos_time:<20} {p.aos_az:>7.1f}   {tca_time:<20} {p.tca_el:>7.1f} {p.tca_az:>7.1f}   {los_time:<20} {p.los_az:>7.1f}")

    print("")
    print("Calculating Doppler for first pass")
    passes[0].calculate_doppler(uplink_freq, downlink_freq)

    # show doppler for first pass
    # need to find a better way
    print(f"AOS Dopper uplink: {passes[0].doppler_up[0]:.1f} Hz")
    print(f"AOS Dopper downlink: {passes[0].doppler_down[0]:.1f} Hz")
    print(f"LOS Dopper uplink: {passes[0].doppler_up[-1]:.1f} Hz")
    print(f"LOS Dopper downlink: {passes[0].doppler_down[-1]:.1f} Hz")
    
    print("")
    print("Calculating plot for first pass")
    passes[0].calculate_path()

    print("Plotting")
    passes[0].plot_polar_flight_path(f"{sat.satName} - {grid} - Polar View")
    print("Complete")

if __name__ == "__main__":
    main()