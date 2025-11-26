use gtk4::gdk::Display;
use gtk4::prelude::*;
use gtk4::{
    Application, ApplicationWindow, Button, DrawingArea, FileChooserAction, FileChooserNative,
    Frame, Label, Orientation, ResponseType, ScrolledWindow, TextBuffer, TextView, gdk,
};
use libshumate::prelude::*;
use plotters::prelude::*;
//use gtk4::glib::clone;
use fitparser::{FitDataRecord, Value, profile::field_types::MesgNum};
use libshumate::{Coordinate, PathLayer, SimpleMap};
use plotters::style::full_palette::BROWN;
use plotters::style::full_palette::CYAN;
use std::fs::File;
use std::io::ErrorKind;

// Only God and I knew what this was doing when I wrote it.
// Know only God knows.

static FRACT_OF_SCREEN: f32 = 0.85;

// Program entry point.
fn main() {
    let app = Application::builder().build();
    app.connect_activate(build_gui);
    app.run();
}

// Calculate the vector average.
fn mean(data: &Vec<f32>) -> f32 {
    let count = data.len();
    // Handle empty data case to prevent division by zero
    if count == 0 {
        return 0.0;
    }
    let sum: f32 = data.iter().sum();
    let mean = sum / (count as f32);
    return mean;
}

// Calculate the vector standard deviation.
fn standard_deviation(data: &Vec<f32>) -> f32 {
    let count = data.len();
    // Handle empty data case to prevent division by zero.
    if count == 0 {
        return 0.0;
    }
    // Calculate the mean (average).
    // .sum() requires an explicit type annotation if not inferred
    let sum: f32 = data.iter().sum();
    let mean = sum / (count as f32);
    // Calculate the variance.
    // Variance is the average of the squared differences from the Mean.
    let squared_differences_sum: f32 = data
        .iter()
        // Map each element to its squared difference from the mean
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        // Sum all the squared differences
        .sum();
    let variance = squared_differences_sum / (count as f32);
    // Standard deviation is the square root of the variance.
    return variance.sqrt();
}

// Find the largest non-NaN in vector, or NaN otherwise:
fn max_vec(vector: Vec<f32>) -> f32 {
    let v = vector.iter().cloned().fold(0. / 0., f32::max);
    return v;
}

// Find the largest non-NaN in vector, or NaN otherwise:
fn min_vec(vector: Vec<f32>) -> f32 {
    let v = vector.iter().cloned().fold(0. / 0., f32::min);
    return v;
}

// Find the plot range values.
fn get_plot_range(data: &Vec<(f32, f32)>) -> (std::ops::Range<f32>, std::ops::Range<f32>) {
    if data.len() == 0 {
        panic!("Can't calculate range. No values supplied.")
    };
    // Split vector of tuples into two vecs
    let (x, y): (Vec<_>, Vec<_>) = data.into_iter().map(|(a, b)| (a, b)).unzip();
    // Find the range of the chart, statistics says 95% should lie between +/3 sigma
    // for a normal distribution.  Let's go with that for the range.
    let _mean_x = mean(&x);
    let mean_y = mean(&y);
    let _sigma_x = standard_deviation(&x);
    let sigma_y = standard_deviation(&y);
    let xrange: std::ops::Range<f32> = min_vec(x.clone())..max_vec(x.clone());
    let yrange: std::ops::Range<f32> = mean_y - 2.0 * sigma_y..mean_y + 2.0 * sigma_y;
    return (xrange, yrange);
}

// Return a session values of "field_name".
fn get_sess_record_field(data: Vec<FitDataRecord>, field_name: &str) -> f64 {
    for item in &data {
        match item.kind() {
            // Individual msgnum::records
            MesgNum::Session => {
                // Retrieve the FitDataField struct.
                for fld in item.fields().iter() {
                    if fld.name() == field_name {
                        return fld.value().clone().try_into().unwrap();
                        //                         println!("{:?}", v64);
                    }
                }
            }
            _ => (), // matches other patterns
        }
    }
    return f64::NAN;
}

// Return a vector of values of "field_name".
fn get_msg_record_field_as_vec(data: Vec<FitDataRecord>, field_name: &str) -> Vec<f64> {
    let mut field_vals: Vec<f64> = Vec::new();
    for item in &data {
        match item.kind() {
            // Individual msgnum::records
            MesgNum::Record => {
                // Retrieve the FitDataField struct.
                for fld in item.fields().iter() {
                    if fld.name() == field_name {
                        let v64: f64 = fld.value().clone().try_into().unwrap();
                        //                         println!("{:?}", v64);
                        field_vals.push(v64);
                    }
                }
            }
            _ => (), // matches other patterns
        }
    }
    return field_vals;
}

// Per Google Gemini AI
/// Helper to convert various numeric Value variants to f64
/*
fn extract_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Float64(v) => Some(*v),
        Value::Float32(v) => Some(*v as f64),
        Value::UInt8(v) => Some(*v as f64),
        Value::UInt16(v) => Some(*v as f64),
        Value::UInt32(v) => Some(*v as f64),
        Value::UInt64(v) => Some(*v as f64),
        Value::SInt8(v) => Some(*v as f64),
        Value::SInt16(v) => Some(*v as f64),
        Value::SInt32(v) => Some(*v as f64),
        Value::SInt64(v) => Some(*v as f64),
        _ => None,
    }
}
*/

// 1. Define a trait for types that can be extracted from a FIT Value
pub trait FromFitValue: Sized {
    fn from_value(v: &Value) -> Option<Self>;
}

// 2. Implement the trait for the types you care about (e.g., f64)
impl FromFitValue for f64 {
    fn from_value(v: &Value) -> Option<Self> {
        match v {
            Value::Float64(f) => Some(*f),
            Value::Float32(f) => Some(*f as f64),
            Value::UInt8(u) => Some(*u as f64),
            Value::UInt16(u) => Some(*u as f64),
            Value::UInt32(u) => Some(*u as f64),
            Value::SInt8(i) => Some(*i as f64),
            Value::SInt16(i) => Some(*i as f64),
            Value::SInt32(i) => Some(*i as f64),
            _ => None, // Handle non-numeric types by ignoring them
        }
    }
}

// 3. Implement for other types (e.g., i64) as needed
impl FromFitValue for i64 {
    fn from_value(v: &Value) -> Option<Self> {
        match v {
            Value::SInt64(i) => Some(*i),
            Value::SInt32(i) => Some(*i as i64),
            Value::UInt32(u) => Some(*u as i64),
            Value::UInt8(u) => Some(*u as i64),
            _ => None,
        }
    }
}

// 4. The Generic Helper Function
pub fn extract_vec<T: FromFitValue>(value: &Value) -> Vec<T> {
    match value {
        // Case A: Standard Array of Values
        Value::Array(arr) => arr.iter().filter_map(T::from_value).collect(),

        // Case B: Byte Array (Special optimization in FIT files)
        // Value::Byte(bytes) => {
        //     // We treat bytes as UInt8 scalars to convert them to T
        //     bytes
        //         .iter()
        //         .filter_map(|&b| T::from_value(&Value::UInt8(b)))
        //         .collect()
        // }

        // Case C: Scalar (Single value treated as 1-item array)
        scalar => {
            if let Some(val) = T::from_value(scalar) {
                vec![val]
            } else {
                Vec::new()
            }
        }
    }
}

// Return a time for heart rate in the time_in_zone record.
fn get_time_in_zone_field(data: &Vec<FitDataRecord>) -> (Option<Vec<f64>>, Option<Vec<f64>>) {
    //fn get_time_in_zone_field(data: &Vec<FitDataRecord>) {
    let mut result: (Option<Vec<f64>>, Option<Vec<f64>>) = (None, None);
    for item in data {
        match item.kind() {
            // Individual msgnum::records
            MesgNum::TimeInZone => {
                // Retrieve the FitDataField struct.
                for fld in item.fields().iter() {
                    if fld.name() == "reference_mesg" && fld.value().to_string() == "session" {
                        let floats: Vec<f64> = extract_vec(item.fields()[2].value());
                        let hr_limits: Vec<f64> = extract_vec(item.fields()[3].value());
                        //println!("{:?}", floats);
                        result = (Some(floats), Some(hr_limits));
                    }
                }
            }
            _ => (), // matches other patterns
        }
    }
    return result;
}

// Convert speed (m/s) to pace(min/mile)
fn cvt_pace(speed: f32) -> f32 {
    if speed < 1.00 {
        return 26.8224; //avoid divide by zero
    } else {
        return 26.8224 / speed;
    }
}

// Convert distance meters to miles.
fn cvt_distance(distance: f32) -> f32 {
    return distance * 0.00062137119;
}

// Convert altitude meters to feet.
fn cvt_altitude(altitude: f32) -> f32 {
    return altitude * 3.2808399;
}

// Convert temperature deg C  to deg F.
fn cvt_temperature(temperature: f32) -> f32 {
    return temperature * 1.8 + 32.0;
}

// Convert semi-circles to degrees.
fn semi_to_degrees(semi: f32) -> f64 {
    let factor: f64 = 2i64.pow(31u32) as f64;
    let deg_val: f64 = semi as f64 * 180f64 / factor;
    return deg_val;
}

// Convert elapsed time in secs to hr,:min,sec.
fn cvt_elapsed_time(time_in_sec: f32) -> (i32, i32, i32) {
    let t = time_in_sec / 3600.0;
    let hr = t.trunc();
    let minsec = t.fract() * 60.0;
    let min = minsec.trunc();
    let s = minsec.fract() * 60.0;
    let sec = s.trunc();
    return (hr as i32, min as i32, sec as i32);
}

// Retrieve converted values to plot from fit file.
fn get_xy(data: &Vec<FitDataRecord>, x_field_name: &str, y_field_name: &str) -> Vec<(f32, f32)> {
    let mut x_user: Vec<f32> = Vec::new();
    let mut y_user: Vec<f32> = Vec::new();
    let mut xy_pairs: Vec<(f32, f32)> = Vec::new();
    // Parameter can be distance, heart_rate, enhanced_speed, enhanced_altitude.
    let x: Vec<f64> = get_msg_record_field_as_vec(data.clone(), x_field_name);
    let y: Vec<f64> = get_msg_record_field_as_vec(data.clone(), y_field_name);
    //  Convert values to 32 bit and create a tuple.
    if (x.len() == y.len()) && (x.len() != 0) && (y.len() != 0) {
        for index in 0..x.len() - 1 {
            match x_field_name {
                "distance" => {
                    x_user.push(cvt_distance(x[index] as f32));
                }
                "enhanced_speed" => {
                    x_user.push(cvt_pace(x[index] as f32));
                }
                "altitude" => {
                    x_user.push(cvt_altitude(x[index] as f32));
                }
                "temperature" => {
                    x_user.push(cvt_temperature(x[index] as f32));
                }
                _ => {
                    x_user.push(x[index] as f32);
                }
            }
        }
    }
    if (x.len() == y.len()) && (x.len() != 0) && (y.len() != 0) {
        for index in 0..y.len() - 1 {
            match y_field_name {
                "distance" => {
                    y_user.push(cvt_distance(y[index] as f32));
                }
                "enhanced_speed" => {
                    y_user.push(cvt_pace(y[index] as f32));
                }
                "altitude" => {
                    y_user.push(cvt_altitude(y[index] as f32));
                }
                "temperature" => {
                    y_user.push(cvt_temperature(y[index] as f32));
                }
                _ => {
                    y_user.push(y[index] as f32);
                }
            }
        }
    }
    if (x_user.len() == y_user.len()) && (x_user.len() != 0) && (y_user.len() != 0) {
        for index in 0..x.len() - 1 {
            xy_pairs.push((x_user[index], y_user[index]));
        }
    }
    return xy_pairs;
}

// Build drawing area.
fn build_da(data: &Vec<FitDataRecord>) -> DrawingArea {
    let drawing_area: DrawingArea = DrawingArea::builder().build();
    // Need to clone to use inside the closure.
    let d = data.clone();
    // Use a "closure" (anonymous function?) as the drawing area draw_func.
    // The pd struct is passed in.
    drawing_area.set_draw_func(move |_drawing_area, cr, width, height| {
        //        println!("{:?}", d);
        // --- 🎨 Custom Drawing Logic Starts Here ---
        let root = plotters_cairo::CairoBackend::new(
            &cr,
            (width.try_into().unwrap(), height.try_into().unwrap()),
        )
        .unwrap()
        .into_drawing_area();
        let _ = root.fill(&WHITE);
        let areas = root.split_evenly((2, 3));
        // Declare and initialize.
        let num_formatter = |x: &f32| format!("{:.3}", x);
        let pace_formatter = |x: &f32| {
            let mins = x.trunc();
            let secs = x - mins;
            format!("{:02.0}:{:02.0}", mins, secs)
        };
        let mut plotvals: Vec<(f32, f32)> = Vec::new();
        let mut caption: &str = "";
        let mut xlabel: &str = "";
        let mut ylabel: &str = "";
        let mut plot_range: (std::ops::Range<f32>, std::ops::Range<f32>) =
            (0_f32..1_f32, 0_f32..1_f32);
        let mut y_formatter: Box<dyn Fn(&f32) -> String> = Box::new(num_formatter);
        let mut color = &RED;

        for (a, idx) in areas.iter().zip(1..) {
            //let root = root.margin(50, 50, 50, 50);
            // After this point, we should be able to construct a chart context
            if idx == 1 {
                plotvals = get_xy(&d, "distance", "enhanced_altitude");
                if plotvals.len() == 0 {
                    continue;
                }
                plot_range = get_plot_range(&plotvals.clone());
                y_formatter = Box::new(num_formatter);
                caption = "Elevation";
                ylabel = "Elevation(feet)";
                xlabel = "Distance(miles)";
                color = &RED;
            }
            if idx == 2 {
                plotvals = get_xy(&d, "distance", "heart_rate");
                if plotvals.len() == 0 {
                    continue;
                }
                plot_range = get_plot_range(&plotvals.clone());
                y_formatter = Box::new(num_formatter);
                caption = "Heart rate";
                ylabel = "Heart rate(bpm)";
                xlabel = "Distance(miles)";
                color = &BLUE;
            }
            if idx == 3 {
                plotvals = get_xy(&d, "distance", "cadence");
                if plotvals.len() == 0 {
                    continue;
                }
                plot_range = get_plot_range(&plotvals.clone());
                y_formatter = Box::new(num_formatter);
                caption = "Cadence";
                ylabel = "Cadence";
                xlabel = "Distance(miles)";
                color = &CYAN;
            }
            if idx == 4 {
                plotvals = get_xy(&d, "distance", "enhanced_speed");
                if plotvals.len() == 0 {
                    continue;
                }
                plot_range = get_plot_range(&plotvals.clone());
                y_formatter = Box::new(pace_formatter);
                caption = "Pace";
                ylabel = "Pace(min/mile)";
                xlabel = "Distance(miles)";
                color = &GREEN;
            }
            if idx == 5 {
                plotvals = get_xy(&d, "distance", "temperature");
                if plotvals.len() == 0 {
                    continue;
                }
                plot_range = get_plot_range(&plotvals.clone());
                y_formatter = Box::new(num_formatter);
                caption = "Temperature";
                ylabel = "Temperature (°F)";
                xlabel = "Distance(miles)";
                color = &BROWN;
            }
            if idx == 6 {
                break;
            }
            let mut chart = ChartBuilder::on(&a)
                // Set the caption of the chart
                .caption(caption, ("sans-serif", 16).into_font())
                // Set the size of the label region
                .x_label_area_size(40)
                .y_label_area_size(60)
                // Finally attach a coordinate on the drawing area and make a chart context
                .build_cartesian_2d(plot_range.clone().0, plot_range.clone().1)
                .unwrap();
            let _ = chart
                .configure_mesh()
                // We can customize the maximum number of labels allowed for each axis
                .x_labels(5)
                .y_labels(5)
                .x_desc(xlabel)
                .y_desc(ylabel)
                .y_label_formatter(&y_formatter)
                .draw();
            // // And we can draw something in the drawing area
            // We need to clone plotvals each time we make a call to LineSeries and PointSeries
            let _ = chart.draw_series(LineSeries::new(plotvals.clone(), color));
        }
        let _ = root.present();
        // --- Custom Drawing Logic Ends Here ---
    }); // --- End closure. 
    return drawing_area;
}

// Adds a PathLayer with a path of given coordinates to the map.
fn add_path_layer_to_map(map: &SimpleMap, path_points: Vec<(f32, f32)>) {
    // Define the RGBA color using the builder pattern for gtk4::gdk::RGBA
    let blue = gdk::RGBA::parse("blue").expect("Failed to parse color");
    let viewport = map.viewport().expect("No viewport.");
    let path_layer = PathLayer::new(&viewport);
    path_layer.set_stroke_color(Some(&blue));
    path_layer.set_stroke_width(3.0); // Thickness in pixels
    for (lat, lon) in path_points {
        let coord = Coordinate::new_full(semi_to_degrees(lat), semi_to_degrees(lon));
        path_layer.add_node(&coord);
    }
    // Add the layer to the map
    map.add_overlay_layer(&path_layer);
}

// Build the map.
fn build_map(data: &Vec<FitDataRecord>) -> SimpleMap {
    let map = SimpleMap::new();
    let source = libshumate::MapSourceRegistry::with_defaults()
        .by_id("osm-mapnik")
        .expect("Could not retrieve map source.");
    map.set_map_source(Some(&source));
    // Get values from fit file.
    let run_path = get_xy(&data, "position_lat", "position_long");
    // Call the function to add the path layer
    add_path_layer_to_map(&map, run_path);
    let viewport = map.viewport().expect("Couldn't get viewport.");
    // You may want to set an initial center and zoom level.
    let nec_lat = get_sess_record_field(data.clone(), "nec_lat");
    let nec_long = get_sess_record_field(data.clone(), "nec_long");
    let swc_lat = get_sess_record_field(data.clone(), "swc_lat");
    let swc_long = get_sess_record_field(data.clone(), "swc_long");
    if !nec_lat.is_nan() & !nec_long.is_nan() & !swc_lat.is_nan() & !swc_long.is_nan() {
        let center_lat = (semi_to_degrees(nec_lat as f32) + semi_to_degrees(swc_lat as f32)) / 2.0;
        let center_long =
            (semi_to_degrees(nec_long as f32) + semi_to_degrees(swc_long as f32)) / 2.0;
        // println!("{:?}", center_lat);
        // println!("{:?}", center_long);
        viewport.set_location(center_lat, center_long);
    } else {
        viewport.set_location(29.7601, -95.3701); // e.g. Houston, USA
    }
    viewport.set_zoom_level(14.0);
    return map;
}

// Build the map.
fn build_summary(data: &Vec<FitDataRecord>, text_buffer: &TextBuffer) {
    text_buffer.set_text("File loaded.");
    // Clear out anything in the buffer.
    let mut start = text_buffer.start_iter();
    let mut end = text_buffer.end_iter();
    text_buffer.delete(&mut start, &mut end);
    let mut lap_index: u8 = 0;
    let mut lap_str: String;
    for item in data {
        match item.kind() {
            MesgNum::Session | MesgNum::Lap => {
                // print all the data records in FIT file
                //println!("{:#?}", item.fields());
                if item.kind() == MesgNum::Session {
                    text_buffer.insert(&mut end, "\n");
                    text_buffer.insert(
                        &mut end,
                        "============================ Session ==================================\n",
                    );
                    text_buffer.insert(&mut end, "\n");
                }
                if item.kind() == MesgNum::Lap {
                    lap_index = lap_index + 1;
                    lap_str = format!(
                        "------------------------------ Lap {}-----------------------------------\n",
                        lap_index
                    );
                    text_buffer.insert(&mut end, "\n");
                    text_buffer.insert(&mut end, &lap_str);
                    text_buffer.insert(&mut end, "\n");
                }
                // Retrieve the FitDataField struct.
                for fld in item.fields().iter() {
                    match fld.name() {
                        "start_position_lat"
                        | "start_position_long"
                        | "end_position_lat"
                        | "end_position_long" => {
                            let semi: i64 = fld.value().try_into().expect("conversion failed"); //semicircles
                            let degrees = semi_to_degrees(semi as f32);
                            let value_str = format!("{:<40}: {degrees:<6.3}°\n", fld.name(),);
                            text_buffer.insert(&mut end, &value_str);
                        }

                        "total_strides"
                        | "total_calories"
                        | "avg_heart_rate"
                        | "max_heart_rate"
                        | "avg_running_cadence"
                        | "max_running_cadence"
                        | "total_training_effect"
                        | "first_lap_index"
                        | "num_laps"
                        | "avg_fractional_cadence"
                        | "max_fractional_cadence"
                        | "total_anaerobic_training_effect"
                        | "sport"
                        | "sub_sport"
                        | "timestamp"
                        | "start_time" => {
                            let value_str = format!(
                                "{:<40}: {:<#} {:<}\n",
                                fld.name(),
                                fld.value(),
                                fld.units()
                            );
                            text_buffer.insert(&mut end, &value_str);
                        }
                        "total_ascent" | "total_descent" => {
                            let val: f64 = fld.value().clone().try_into().unwrap();
                            let val_cvt = cvt_altitude(val as f32);
                            let value_str =
                                format!("{:<40}: {:<.2} {:<}\n", fld.name(), val_cvt, "feet");
                            text_buffer.insert(&mut end, &value_str);
                        }
                        "total_distance" => {
                            let val: f64 = fld.value().clone().try_into().unwrap();
                            let val_cvt = cvt_distance(val as f32);
                            let value_str =
                                format!("{:<40}: {:<.2} {:<}\n", fld.name(), val_cvt, "miles");
                            text_buffer.insert(&mut end, &value_str);
                        }
                        "total_elapsed_time" | "total_timer_time" => {
                            let val: f64 = fld.value().clone().try_into().unwrap();
                            let val_cvt = cvt_elapsed_time(val as f32);
                            let value_str = format!(
                                "{:<40}: {:01}h:{:02}m:{:02}s\n",
                                fld.name(),
                                val_cvt.0,
                                val_cvt.1,
                                val_cvt.2
                            );
                            text_buffer.insert(&mut end, &value_str);
                        }
                        "min_temperature" | "max_temperature" | "avg_temperature" => {
                            let val: i64 = fld.value().try_into().expect("conversion failed");
                            let val_cvt = cvt_temperature(val as f32);
                            let value_str =
                                format!("{:<40}: {:<.2} {:<}\n", fld.name(), val_cvt, "°F");
                            text_buffer.insert(&mut end, &value_str);
                        }
                        "enhanced_avg_speed" | "enhanced_max_speed" => {
                            let val: f64 = fld.value().clone().try_into().unwrap();
                            let val_cvt = cvt_pace(val as f32);
                            let value_str =
                                format!("{:<40}: {:<.2} {:<}\n", fld.name(), val_cvt, "min/mile");
                            text_buffer.insert(&mut end, &value_str);
                        }
                        _ => print!("{}", ""), // matches other patterns
                    }
                }
            }
            _ => print!("{}", ""), // matches other patterns
        }
    }
    if let (Some(zone_times), Some(zone_limits)) = get_time_in_zone_field(data) {
        // There are 7 zones but only 6 upper limits.
        text_buffer.insert(&mut end, "\n");
        text_buffer.insert(
            &mut end,
            "=================== Time in Heart Rate Zones for Session  ========\n",
        );
        for (z, val) in zone_times.iter().enumerate() {
            let val_cvt = cvt_elapsed_time(*val as f32);
            let mut ll: f64;
            let mut ul: f64;
            if z == 0 {
                ll = 0.0;
                ul = zone_limits[z];
            } else if z < zone_limits.len() && z > 0 {
                // ll = zone_limits[z];
                // ul = zone_limits[z - 1];
                ll = zone_limits[z - 1];
                ul = zone_limits[z];
            } else {
                ll = zone_limits[z - 1];
                ul = 220.0;
            }
            let value_str = format!(
                "{:<5}{:<35}({:?}-{:?} bpm): {:01}h:{:02}m:{:02}s\n",
                "Zone", z, ll, ul, val_cvt.0, val_cvt.1, val_cvt.2
            );
            text_buffer.insert(&mut end, &value_str);
        }
        text_buffer.insert(&mut end, "\n");
    };
}

// Find out how many pixels we have to work with.
fn get_geometry() -> (i32, i32) {
    // 1. Get the default display (connection to the window server)
    if let Some(display) = Display::default() {
        // 2. Get the list of monitors (returns a ListModel)
        let monitors = display.monitors();

        // 3. We typically want the first monitor (primary)
        // Note: In a multi-monitor setup, you might want to find which monitor
        // the window is actually on, but during "build_ui", the window isn't visible yet.
        if let Some(monitor_obj) = monitors.item(0) {
            // Downcast the generic object to a GdkMonitor
            if let Ok(monitor) = monitor_obj.downcast::<gdk::Monitor>() {
                // 4. Get geometry (x, y, width, height)
                let geometry = monitor.geometry();
                return (geometry.width(), geometry.height());
            }
        }
    }
    return (1024, 768);
}
// Create the GUI.
fn build_gui(app: &Application) {
    let (width, height) = get_geometry();
    let app_width = FRACT_OF_SCREEN * width as f32;
    let app_height = FRACT_OF_SCREEN * height as f32;
    let win = ApplicationWindow::builder()
        .application(app)
        .default_width(app_width.trunc() as i32)
        .default_height(app_height.trunc() as i32)
        .title("Test")
        .build();

    let outer_box = gtk4::Box::new(Orientation::Vertical, 10);
    // Main horizontal container to hold the two frames side-by-side
    let main_box = gtk4::Box::new(Orientation::Horizontal, 10);
    let inner_box = gtk4::Box::new(Orientation::Vertical, 10);
    let text_view = TextView::builder().build();
    text_view.set_monospace(true);
    let text_buffer = text_view.buffer();

    main_box.set_vexpand(true);
    main_box.set_hexpand(true);
    let frame_left = Frame::builder().build();
    let frame_right = Frame::builder().build();
    let btn = Button::with_label("Select a file...");
    let label_path = Label::new(Some("No file selected"));

    let frame_left_handle = frame_left.clone();
    let frame_right_handle = frame_right.clone();
    let window_clone = win.clone();
    let label_clone = label_path.clone();
    let text_buffer_handle = text_buffer.clone();

    btn.connect_clicked(move |_| {
        // 1. Create the Native Dialog
        // Notice the arguments: Title, Parent Window, Action, Accept Label, Cancel Label
        let native = FileChooserNative::new(
            Some("Open File Native"),
            Some(&window_clone),
            FileChooserAction::Open,
            Some("Open"),   // Custom label for the "OK" button
            Some("Cancel"), // Custom label for the "Cancel" button
        );

        // We need another clone of the label for the dialog's internal closure
        let label_for_dialog = label_clone.clone();
        let frame_left_handle2 = frame_left_handle.clone();
        let frame_right_handle2 = frame_right_handle.clone();
        let text_buffer_handle2 = text_buffer_handle.clone();

        // 2. Connect to the response signal
        native.connect_response(move |dialog, response| {
            if response == ResponseType::Accept {
                // Extract the file path
                if let Some(file) = dialog.file() {
                    if let Some(path) = file.path() {
                        let path_str = path.to_string_lossy();
                        label_for_dialog.set_text(&path_str);
                        // Get values from fit file.
                        let file_result = File::open(&*path_str);
                        let mut file = match file_result {
                            Ok(file) => file,
                            Err(error) => match error.kind() {
                                // Handle specifically "Not Found"
                                ErrorKind::NotFound => {
                                    panic!("File not found.");
                                }
                                _ => {
                                    panic!("Hmmm...unknown error. Check file permissions?");
                                }
                            },
                        };
                        // Read the fit file and create the map and graph drawing area.
                        if let Ok(data) = fitparser::from_reader(&mut file) {
                            let shumate_map = build_map(&data);
                            frame_left_handle2.set_child(Some(&shumate_map));
                            let da = build_da(&data);
                            frame_right_handle2.set_child(Some(&da));
                            build_summary(&data, &text_buffer_handle2);
                        }
                    }
                }
            } else {
                println!("User cancelled");
            }
            // unlike FileChooserDialog, 'native' creates a transient reference.
            // It's good practice to drop references, but GTK handles the cleanup
            // once it goes out of scope or the window closes.
        });

        // 3. Show the dialog
        native.show();
    });

    // Inner box contains only the map and text summary
    inner_box.append(&frame_left);
    inner_box.set_homogeneous(true);
    // TextViews do not scroll by default; they must be wrapped in a ScrolledWindow.
    let scrolled_window = ScrolledWindow::builder()
        //        .hscrollbar_policy(gtk::PolicyType::Never) // Disable horizontal scrolling
        .min_content_width(300)
        .min_content_height(200)
        .child(&text_view)
        .build();
    inner_box.append(&scrolled_window);
    // Main box contains all of the above plus the graphs.
    main_box.append(&inner_box);
    main_box.append(&frame_right);
    main_box.set_homogeneous(true); // Ensures both frames take exactly half the window width
    // Outer box contains the above and the file load button.
    outer_box.append(&btn);
    outer_box.append(&main_box);
    win.set_child(Some(&outer_box));
    win.present();
}
