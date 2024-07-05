.. _new_orbit_anx:

How to use the new Interpolated Orbit
=====================================


Overview
--------

Arepytools **v1.7.0** brings a new **interpolated Orbit** object and a dedicated **ANX time module** that aim to be
substitute the usage of the old GeneralSarOrbit class, improving code clarity, reduce coupling and ease the sensor's
orbit management.

.. note::

   ``GeneralSarOrbit`` will not be deprecated due to its wide spread and the amount of legacy code. However, it is highly
   recommended to switch to this new implementation at least for new projects and new code.


Orbit object
------------

Creation
~~~~~~~~

The new ``Orbit`` object is based on a Cubic Spline interpolator that let the user interpolate positions, velocities and
accelerations within the time domain boundaries. Extrapolation is not allowed by design.

The ``Orbit`` object can be accessed from the `arepytools.geometry.orbit` module and can be instantiated directly from The
constructor of the class, providing times, positions and velocities arrays, or by using a dedicated ``create_orbit()``
function available in `arepytools.io` module, as shown in the following examples.

.. code-block:: python

    from arepytools import io
	from arepytools.geometry.orbit import Orbit

    # position shape (N, 3)
    sensor_positions = np.array(
        [
            [-2542286.449576481, -5094859.4894666, 3901083.7183820857],
            [-2542066.5079547316, -5092594.238796566, 3904175.2612526673],
            [-2541845.6347068036, -5090327.468904538, 3907265.6186505724],
        ]
    )
    # velocities shape (N, 3)
    sensor_velocities = np.array(
        [
            [439, 4529, 6184],
            [440, 4532, 6181],
            [442, 4535, 6179],
        ]
    )
    # time axis shape (N,)
    DT = 0.5
    time_axis_relative = np.arange(0, 10, DT)
    time_axis = PreciseDateTime.from_utc_string("13-FEB-2023 09:33:56.000000") + time_axis_relative

    # creating orbit from class constructor
    orbit_from_class = Orbit(times=time_axis, positions=sensor_positions, velocities=sensor_velocities)

    # creating orbit from StateVector metadata
    path_to_pf = ...
    product = io.open_product_folder(path_to_pf)
    metadata_first_channel = io.read_metadata(product.get_channel_metadata(1))

    orbit_from_func = io.create_orbit(state_vectors=metadata_first_channel.get_state_vectors())


Methods and Properties
~~~~~~~~~~~~~~~~~~~~~~

The newly created Orbit object can be used to compute sensor position, velocity and acceleration at a given time as long
as that time lies within the object time domain boundaries. The following line of codes shows how to determine the orbit
time boundaries and how to interpolate inside them to get the desired information.

Orbit properties:

.. code-block:: python

    # access positions vector used to instantiate the orbit
    print(orbit.positions)
    # access velocities vector used to instantiate the orbit
    print(orbit.velocities)
    # access time vector used to instantiate the orbit
    print(orbit.times)
    # orbit time domain lower and upper boundaries, it's equal to [orbit.times[0], orbit.times[-1]]
    print(orbit.domain)

Orbit methods:

.. code-block:: python

    from arepytools.timing.precisedatetime import PreciseDateTime

    generic_time = PreciseDateTime.from_utc_string("13-FEB-2023 09:33:57.500000")

    # get interpolated position
    interpolated_position = orbit.evaluate(generic_time)
    # get interpolated velocity
    interpolated_velocity = orbit.evaluate_first_derivatives(generic_time)
    # get interpolated acceleration
    interpolated_acceleration = orbit.evaluate_second_derivatives(generic_time)

    # works with array too
    interpolated_positions = orbit.evaluate(generic_time + np.arange(0, 2, 0.5))


Orbit auxiliary functions
-------------------------

A set of auxiliary functions compatible with the Orbit object has been developed to perform the same operations available
for the old GeneralSarOrbit without embedding those functionalities into the Orbit itself.

For example, **sat2earth** and **earth2sat** methods are not available in the Orbit object and therefore these operations should
be performed using the proper functionalities available in the direct_geocoding and inverse_geocoding modules.

`arepytools.geometry` core functionalities have already been updated to match this new format as input. Also, the new **ANX time module**
accessible via `arepytools.geometry.anx_time` offers the same functionalities to compute anx times once enclosed in the GeneralSarOrbit
definition as standalone free functions.

The following code comparison shows the difference in performing the most common operations between the GeneralSarOrbit
and Orbit objects.

+-------------------------------------+--------------------------------------+
|                                     |                                      |
|.. literalinclude:: old_code_orb.txt |.. literalinclude:: new_code_orb.txt  |
|                                     |                                      |
+-------------------------------------+--------------------------------------+
