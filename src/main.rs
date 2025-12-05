use chrono::Datelike;
use chrono::NaiveDateTime;
use fitparser::{FitDataRecord, Value, profile::field_types::MesgNum};
use gtk4::cairo::Context;
use gtk4::glib::clone;
use gtk4::prelude::*;
use gtk4::{
    Adjustment, Application, ApplicationWindow, Button, DrawingArea, DropDown, FileChooserAction,
    FileChooserNative, Frame, Image, Label, Orientation, ResponseType, Scale, ScrolledWindow,
    StringList, StringObject, TextBuffer, TextView, gdk,
};
use libshumate::prelude::*;
use libshumate::{Coordinate, Marker, MarkerLayer, PathLayer, SimpleMap};
use plotters::prelude::*;
use plotters::style::full_palette::BROWN;
use plotters::style::full_palette::CYAN;
use std::fs::File;
use std::io::ErrorKind;
// Only God and I knew what this was doing when I wrote it.
// Now only God knows.

// Unit of measure system.
enum Units {
    Metric,
    US,
    None,
}

// Program entry point.
fn main() {
    let app = Application::builder().build();
    app.connect_activate(build_gui);
    app.run();
}

fn get_unit_system(units_widget: &DropDown) -> Units {
    if units_widget.model().is_some() {
        let model = units_widget.model().unwrap();
        if let Some(item_obj) = model.item(units_widget.selected()) {
            if let Ok(string_obj) = item_obj.downcast::<StringObject>() {
                let unit_string = String::from(string_obj.string());
                //println!("{:?}", unit_string);
                if unit_string == "🇪🇺 Metric" {
                    return Units::Metric;
                }
                if unit_string == "🇺🇸 US" {
                    return Units::US;
                }
            }
        }
    }
    return Units::None;
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
fn set_plot_range(
    data: &Vec<(f32, f32)>,
    zoom_x: f32,
    zoom_y: f32,
) -> (std::ops::Range<f32>, std::ops::Range<f32>) {
    if data.len() == 0 {
        panic!("Can't calculate range. No values supplied.")
    };
    if (zoom_x < 0.01) | (zoom_y < 0.01) {
        panic!("Invalid zoom.")
    }
    // Split vector of tuples into two vecs
    let (x, y): (Vec<_>, Vec<_>) = data.into_iter().map(|(a, b)| (a, b)).unzip();
    // Find the range of the chart, statistics says 95% should lie between +/3 sigma
    // for a normal distribution.  Let's go with that for the range.
    let _mean_x = mean(&x);
    let mean_y = mean(&y);
    let _sigma_x = standard_deviation(&x);
    let sigma_y = standard_deviation(&y);
    // Disallow zero, negative values of zoom.
    let xrange: std::ops::Range<f32> = min_vec(x.clone())..1.0 / zoom_x * max_vec(x.clone());
    let yrange: std::ops::Range<f32> =
        mean_y - 2.0 / zoom_y * sigma_y..mean_y + 2.0 / zoom_y * sigma_y;
    mean_y - 2.0 / zoom_y * sigma_y..mean_y + 2.0 / zoom_y * sigma_y;
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
            _ => (), // iatches other patterns
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

// Convert speed (m/s) to pace(min/mile, min/km)
fn cvt_pace(speed: f32, units: &Units) -> f32 {
    match units {
        Units::US => {
            if speed < 1.00 {
                return 26.8224; //avoid divide by zero
            } else {
                return 26.8224 / speed;
            }
        }
        Units::Metric => {
            if speed < 1.00 {
                return 16.666667; //avoid divide by zero
            } else {
                return 16.666667 / speed;
            }
        }
        Units::None => {
            return speed;
        }
    }
}

// Convert distance meters to miles, km.
fn cvt_distance(distance: f32, units: &Units) -> f32 {
    match units {
        Units::US => {
            return distance * 0.00062137119;
        }
        Units::Metric => {
            return distance * 0.001;
        }
        Units::None => {
            return distance;
        }
    }
}

// Convert altitude meters to feet, m.
fn cvt_altitude(altitude: f32, units: &Units) -> f32 {
    match units {
        Units::US => {
            return altitude * 3.2808399;
        }
        Units::Metric => {
            return altitude * 1.0;
        }
        Units::None => {
            return altitude;
        }
    }
}

// Convert temperature deg C  to deg F, deg C.
fn cvt_temperature(temperature: f32, units: &Units) -> f32 {
    match units {
        Units::US => {
            return temperature * 1.8 + 32.0;
        }
        Units::Metric => {
            return temperature * 1.0;
        }
        Units::None => {
            return temperature;
        }
    }
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
fn get_xy(
    data: &Vec<FitDataRecord>,
    units_widget: &DropDown,
    x_field_name: &str,
    y_field_name: &str,
) -> Vec<(f32, f32)> {
    let mut x_user: Vec<f32> = Vec::new();
    let mut y_user: Vec<f32> = Vec::new();
    let mut xy_pairs: Vec<(f32, f32)> = Vec::new();
    // Get the enumerated value for the unit system the user selected.
    let user_unit = get_unit_system(units_widget);
    // Parameter can be distance, heart_rate, enhanced_speed, enhanced_altitude.
    let x: Vec<f64> = get_msg_record_field_as_vec(data.clone(), x_field_name);
    let y: Vec<f64> = get_msg_record_field_as_vec(data.clone(), y_field_name);
    //  Convert values to 32 bit and create a tuple.
    // Occasionally we see off by one errors in the data.
    // If true, Chop the last one. Must be careful comparing usize values.
    let mut data_range = 0..0;
    if x.len() != 0 {
        data_range = 0..x.len() - 1;
    }
    if x.len() > y.len() && (x.len() != 0) && (y.len() != 0) {
        data_range = 0..y.len() - 1;
    }
    if (x.len() != 0) && (y.len() != 0) {
        for index in data_range.clone() {
            match x_field_name {
                "distance" => {
                    x_user.push(cvt_distance(x[index] as f32, &user_unit));
                }
                "enhanced_speed" => {
                    x_user.push(cvt_pace(x[index] as f32, &user_unit));
                }
                "enhanced_altitude" => {
                    x_user.push(cvt_altitude(x[index] as f32, &user_unit));
                }
                "temperature" => {
                    x_user.push(cvt_temperature(x[index] as f32, &user_unit));
                }
                _ => {
                    x_user.push(x[index] as f32);
                }
            }
        }
    }
    if (x.len() != 0) && (y.len() != 0) {
        for index in data_range.clone() {
            match y_field_name {
                "distance" => {
                    y_user.push(cvt_distance(y[index] as f32, &user_unit));
                }
                "enhanced_speed" => {
                    y_user.push(cvt_pace(y[index] as f32, &user_unit));
                }
                "enhanced_altitude" => {
                    y_user.push(cvt_altitude(y[index] as f32, &user_unit));
                }
                "temperature" => {
                    y_user.push(cvt_temperature(y[index] as f32, &user_unit));
                }
                _ => {
                    y_user.push(y[index] as f32);
                }
            }
        }
    }
    if (x_user.len() != 0) && (y_user.len() != 0) {
        for index in data_range.clone() {
            xy_pairs.push((x_user[index], y_user[index]));
        }
    }
    return xy_pairs;
}

// Use plotters.rs to draw a graph on the drawing area.
fn draw_graphs(
    d: &Vec<FitDataRecord>,
    units_widget: &DropDown,
    xzm: &Adjustment,
    yzm: &Adjustment,
    curr_adj: &Adjustment,
    cr: &Context,
    width: f64,
    height: f64,
) {
    let zoom_x: f32 = xzm.value() as f32;
    let zoom_y: f32 = yzm.value() as f32;
    let user_unit = get_unit_system(units_widget);
    //        println!("{:?}", d);
    // --- 🎨 Custom Drawing Logic Starts Here ---
    let root = plotters_cairo::CairoBackend::new(&cr, (width as u32, height as u32))
        .unwrap()
        .into_drawing_area();
    let _ = root.fill(&WHITE);
    let areas = root.split_evenly((2, 3));
    // Declare and initialize.
    let num_formatter = |x: &f32| format!("{:7.2}", x);
    let pace_formatter = |x: &f32| {
        let mins = x.trunc();
        let secs = x.fract() * 60.0;
        format!("{:02.0}:{:02.0}", mins, secs)
    };
    let mut plotvals: Vec<(f32, f32)> = Vec::new();
    let mut caption: &str = "";
    let mut xlabel: &str = "";
    let mut ylabel: &str = "";
    let mut plot_range: (std::ops::Range<f32>, std::ops::Range<f32>) = (0_f32..1_f32, 0_f32..1_f32);
    let mut y_formatter: Box<dyn Fn(&f32) -> String> = Box::new(num_formatter);
    let mut color = &RED;
    for (a, idx) in areas.iter().zip(1..) {
        //let root = root.margin(50, 50, 50, 50);
        // After this point, we should be able to construct a chart context
        if idx == 1 {
            plotvals = get_xy(&d, &units_widget, "distance", "enhanced_speed");
            if plotvals.len() == 0 {
                continue;
            }
            plot_range = set_plot_range(&plotvals.clone(), zoom_x, zoom_y);
            y_formatter = Box::new(pace_formatter);
            caption = "Pace";
            match user_unit {
                Units::US => {
                    ylabel = "Pace(min/mile)";
                    xlabel = "Distance(miles)";
                }
                Units::Metric => {
                    ylabel = "Pace(min/km)";
                    xlabel = "Distance(km)";
                }
                Units::None => {
                    ylabel = "";
                    xlabel = "";
                }
            }
            color = &GREEN;
        }
        if idx == 2 {
            plotvals = get_xy(&d, &units_widget, "distance", "heart_rate");
            if plotvals.len() == 0 {
                continue;
            }
            plot_range = set_plot_range(&plotvals.clone(), zoom_x, zoom_y);
            y_formatter = Box::new(num_formatter);
            caption = "Heart rate";
            match user_unit {
                Units::US => {
                    ylabel = "Heart rate(bpm)";
                    xlabel = "Distance(miles)";
                }
                Units::Metric => {
                    ylabel = "Heart rate(bpm)";
                    xlabel = "Distance(km)";
                }
                Units::None => {
                    ylabel = "";
                    xlabel = "";
                }
            }
            color = &BLUE;
        }
        if idx == 3 {
            plotvals = get_xy(&d, &units_widget, "distance", "cadence");
            if plotvals.len() == 0 {
                continue;
            }
            plot_range = set_plot_range(&plotvals.clone(), zoom_x, zoom_y);
            y_formatter = Box::new(num_formatter);
            caption = "Cadence";
            match user_unit {
                Units::US => {
                    ylabel = "Cadence";
                    xlabel = "Distance(miles)";
                }
                Units::Metric => {
                    ylabel = "Cadence";
                    xlabel = "Distance(km)";
                }
                Units::None => {
                    ylabel = "";
                    xlabel = "";
                }
            }
            color = &CYAN;
        }
        if idx == 4 {
            plotvals = get_xy(&d, &units_widget, "distance", "enhanced_altitude");
            if plotvals.len() == 0 {
                continue;
            }
            plot_range = set_plot_range(&plotvals.clone(), zoom_x, zoom_y);
            y_formatter = Box::new(num_formatter);
            caption = "Elevation";
            match user_unit {
                Units::US => {
                    ylabel = "Elevation(feet)";
                    xlabel = "Distance(miles)";
                }
                Units::Metric => {
                    ylabel = "Elevation(m)";
                    xlabel = "Distance(km)";
                }
                Units::None => {
                    ylabel = "";
                    xlabel = "";
                }
            }
            color = &RED;
        }
        if idx == 5 {
            plotvals = get_xy(&d, &units_widget, "distance", "temperature");
            if plotvals.len() == 0 {
                continue;
            }
            plot_range = set_plot_range(&plotvals.clone(), zoom_x, zoom_y);
            y_formatter = Box::new(num_formatter);
            caption = "Temperature";
            match user_unit {
                Units::US => {
                    ylabel = "Temperature(°F)";
                    xlabel = "Distance(miles)";
                }
                Units::Metric => {
                    ylabel = "Temperature(°C)";
                    xlabel = "Distance(km)";
                }
                Units::None => {
                    ylabel = "";
                    xlabel = "";
                }
            }
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
            .margin(10)
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
        // Calculate the hairline.
        let idx = (curr_adj.value() * (plotvals.len() as f64 - 1.0)).trunc() as usize;
        if idx > 0 && idx < plotvals.len() - 1 {
            let hair_x = plotvals[idx].0;
            let hair_y = plotvals[idx].1;
            let mylabel = format!(
                "{:<1}: {:<5.2}{:<1}: {:>1}",
                xlabel,
                hair_x,
                ylabel,
                &y_formatter(&hair_y)
            )
            .to_string();
            let hair_y_min = plot_range.clone().0.start;
            let hair_y_max = plot_range.clone().1.end;
            let mut hairlinevals: Vec<(f32, f32)> = Vec::new();
            hairlinevals.push((hair_x, hair_y_min));
            hairlinevals.push((hair_x, hair_y_max));
            let _ = chart
                .draw_series(DashedLineSeries::new(
                    hairlinevals,
                    1,
                    4,
                    ShapeStyle {
                        color: BLACK.mix(1.0),
                        filled: false,
                        stroke_width: 1,
                    },
                ))
                .unwrap()
                .label(mylabel);

            chart
                .configure_series_labels()
                .position(SeriesLabelPosition::UpperLeft)
                .margin(5)
                .legend_area_size(0)
                .label_font(("Calibri", 10))
                .draw()
                .unwrap();
        }
    }
    let _ = root.present();
    // --- Custom Drawing Logic Ends Here ---
}

// Build drawing area.
fn build_da(
    data: &Vec<FitDataRecord>,
    units_widget: &DropDown,
) -> (DrawingArea, Adjustment, Adjustment, Adjustment) {
    let drawing_area: DrawingArea = DrawingArea::builder().build();
    // Need to clone to use inside the closure.
    let d = data.clone();

    let yzm = Adjustment::builder()
        // The minimum value
        .lower(0.2)
        // The maximum value
        .upper(4.0)
        // Small step increment (for arrow keys/buttons)
        .step_increment(0.1)
        // Large step increment (for Page Up/Page Down keys)
        .page_increment(0.2)
        // The size of the viewable area (not often used for SpinButton, usually 0.0)
        .page_size(0.0)
        .build();
    yzm.set_value(1.0);

    let xzm = Adjustment::builder()
        // The minimum value
        .lower(0.5)
        // The maximum value
        .upper(2.0)
        // Small step increment (for arrow keys/buttons)
        .step_increment(0.1)
        // Large step increment (for Page Up/Page Down keys)
        .page_increment(0.2)
        // The size of the viewable area (not often used for SpinButton, usually 0.0)
        .page_size(0.0)
        .build();
    xzm.set_value(1.0);

    // Represents a normalized fraction of the run.
    let curr_pos = Adjustment::builder()
        // The minimum value
        .lower(0.0)
        // The maximum value
        .upper(1.0)
        // Small step increment (for arrow keys/buttons)
        .step_increment(0.05)
        // Large step increment (for Page Up/Page Down keys)
        .page_increment(0.1)
        // The size of the viewable area (not often used for SpinButton, usually 0.0)
        .page_size(0.0)
        .build();
    curr_pos.set_value(0.001);

    let x_zoom = xzm.clone();
    let y_zoom = yzm.clone();
    let pos = curr_pos.clone();
    let units_clone = units_widget.clone();
    drawing_area.set_draw_func(move |_drawing_area, cr, width, height| {
        draw_graphs(
            &d,
            &units_clone,
            &x_zoom,
            &y_zoom,
            &curr_pos,
            cr,
            width as f64,
            height as f64,
        );
    });
    return (drawing_area, xzm, yzm, pos);
}

// Add a marker layer to the map.
fn add_marker_layer_to_map(map: &SimpleMap) -> Option<MarkerLayer> {
    if map.viewport().is_some() {
        let viewport = map.viewport().unwrap();
        let marker_layer = libshumate::MarkerLayer::new(&viewport);
        map.add_overlay_layer(&marker_layer);
        return Some(marker_layer.clone());
    }
    return None;
}

//Adds a PathLayer with a path of given coordinates to the map.
fn add_path_layer_to_map(map: &SimpleMap, path_points: Vec<(f32, f32)>) {
    if map.viewport().is_some() {
        let viewport = map.viewport().unwrap();
        let path_layer = PathLayer::new(&viewport);
        let result = gdk::RGBA::parse("blue");
        match result {
            Ok(_) => {
                let blue = gdk::RGBA::parse("blue").unwrap();
                path_layer.set_stroke_color(Some(&blue));
            }
            Err(_) => {}
        }
        path_layer.set_stroke_width(2.0); // Thickness in pixels
        for (lat, lon) in path_points {
            let coord = Coordinate::new_full(semi_to_degrees(lat), semi_to_degrees(lon));
            path_layer.add_node(&coord);
        }
        // Add the layer to the map
        map.add_overlay_layer(&path_layer);
    }
}

// Helper function to return the date a run started on.
fn get_run_start_date(data: &Vec<FitDataRecord>) -> (i32, u32, u32) {
    let mut month = 0;
    let mut day = 0;
    let mut year = 0;
    for item in data {
        match item.kind() {
            MesgNum::Session => {
                for fld in item.fields().iter() {
                    if fld.name() == "start_time" {
                        let time_stamp = fld.value().clone().to_string();
                        let from: Result<NaiveDateTime, chrono::ParseError> =
                            NaiveDateTime::parse_from_str(&time_stamp, "%Y-%m-%d %H:%M:%S %z");
                        match from {
                            Ok(date_time) => {
                                year = date_time.date().year();
                                month = date_time.date().month();
                                day = date_time.date().day();
                            }
                            Err(_e) => {
                                panic!("Couldn't parse time_stamp.");
                            }
                        };
                    }
                }
            }
            _ => {}
        }
    }
    return (year, month, day);
}

fn get_symbol(data: &Vec<FitDataRecord>) -> &str {
    //    let mut symbol = "🏃";
    let mut symbol = concat!(r#"<span size="200%">"#, "🏃", "</span>");
    let (_year, month, day) = get_run_start_date(data);
    if month == 1 && day == 1 {
        symbol = concat!(r#"<span size="200%">"#, "🍾", "</span>");
    }
    if month == 3 && day == 17 {
        symbol = concat!(r#"<span size="200%">"#, "🍀", "</span>");
    }
    if month == 7 && day == 4 {
        symbol = concat!(r#"<span size="200%">"#, "🎆", "</span>");
    }
    if month == 10 && day == 31 {
        symbol = concat!(r#"<span size="200%">"#, "🎃", "</span>");
    }
    if month == 12 && day == 24 {
        symbol = concat!(r#"<span size="200%">"#, "🎅", "</span>");
    }
    if month == 12 && day == 25 {
        symbol = concat!(r#"<span size="200%">"#, "🎁", "</span>");
    }
    if month == 12 && day == 31 {
        symbol = concat!(r#"<span size="200%">"#, "🍾", "</span>");
    }
    let _ = "📍";
    return symbol;
}
// Build the map.
fn build_map(data: &Vec<FitDataRecord>) -> (Option<SimpleMap>, Option<MarkerLayer>) {
    if libshumate::MapSourceRegistry::with_defaults()
        .by_id("osm-mapnik")
        .is_some()
    {
        let source = libshumate::MapSourceRegistry::with_defaults()
            .by_id("osm-mapnik")
            .unwrap();
        let map = SimpleMap::new();
        map.set_map_source(Some(&source));
        // Get values from fit file.
        let units_widget = DropDown::builder().build(); // bogus value - no units required for position
        let run_path = get_xy(&data, &units_widget, "position_lat", "position_long");
        // Call the function to add the path layer
        add_path_layer_to_map(&map, run_path.clone());
        // add pins for the starting and stopping points of the run
        let startstop_layer = add_marker_layer_to_map(&map);
        let len = run_path.len();
        if len > 0 {
            let start_lat_deg = semi_to_degrees(run_path[0..1][0].0);
            let start_lon_deg = semi_to_degrees(run_path[0..1][0].1);
            let stop_lat_deg = semi_to_degrees(run_path[len - 1..len][0].0);
            let stop_lon_deg = semi_to_degrees(run_path[len - 1..len][0].1);
            let start_content = gtk4::Label::new(Some("🟢"));
            let stop_content = gtk4::Label::new(Some("🔴"));
            start_content.set_halign(gtk4::Align::Center);
            start_content.set_valign(gtk4::Align::Baseline);
            stop_content.set_halign(gtk4::Align::Center);
            stop_content.set_valign(gtk4::Align::Baseline);
            let start_widget = &start_content;
            let stop_widget = &stop_content;
            let start_marker = Marker::builder()
                .latitude(start_lat_deg)
                .longitude(start_lon_deg)
                .child(&start_widget.clone())
                // Set the visual content widget
                .build();
            let stop_marker = Marker::builder()
                .latitude(stop_lat_deg)
                .longitude(stop_lon_deg)
                .child(&stop_widget.clone())
                // Set the visual content widget
                .build();
            if startstop_layer.is_some() {
                startstop_layer.as_ref().unwrap().add_marker(&start_marker);
                startstop_layer.as_ref().unwrap().add_marker(&stop_marker);
            }
        }
        let marker_layer = add_marker_layer_to_map(&map);
        // You may want to set an initial center and zoom level.
        if map.viewport().is_some() {
            let viewport = map.viewport().unwrap();
            let nec_lat = get_sess_record_field(data.clone(), "nec_lat");
            let nec_long = get_sess_record_field(data.clone(), "nec_long");
            let swc_lat = get_sess_record_field(data.clone(), "swc_lat");
            let swc_long = get_sess_record_field(data.clone(), "swc_long");
            if !nec_lat.is_nan() & !nec_long.is_nan() & !swc_lat.is_nan() & !swc_long.is_nan() {
                let center_lat =
                    (semi_to_degrees(nec_lat as f32) + semi_to_degrees(swc_lat as f32)) / 2.0;
                let center_long =
                    (semi_to_degrees(nec_long as f32) + semi_to_degrees(swc_long as f32)) / 2.0;
                // println!("{:?}", center_lat);
                // println!("{:?}", center_long);
                viewport.set_location(center_lat, center_long);
            } else {
                viewport.set_location(29.7601, -95.3701); // e.g. Houston, USA
            }
            viewport.set_zoom_level(14.0);
        }
        return (Some(map), marker_layer);
    }
    return (None, None); // Can't find map source. Check internet access?
}

// Build the map.
fn build_summary(data: &Vec<FitDataRecord>, units_widget: &DropDown, text_buffer: &TextBuffer) {
    // Get the enumerated value for the unit system the user selected.
    let user_unit = get_unit_system(units_widget);
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
                            let val_cvt = cvt_altitude(val as f32, &user_unit);
                            match user_unit {
                                Units::US => {
                                    let value_str = format!(
                                        "{:<40}: {:<.2} {:<}\n",
                                        fld.name(),
                                        val_cvt,
                                        "feet"
                                    );
                                    text_buffer.insert(&mut end, &value_str);
                                }
                                Units::Metric => {
                                    let value_str = format!(
                                        "{:<40}: {:<.2} {:<}\n",
                                        fld.name(),
                                        val_cvt,
                                        "meters"
                                    );
                                    text_buffer.insert(&mut end, &value_str);
                                }
                                Units::None => {
                                    let value_str =
                                        format!("{:<40}: {:<.2} {:<}\n", fld.name(), val_cvt, "");
                                    text_buffer.insert(&mut end, &value_str);
                                }
                            }
                        }
                        "total_distance" => {
                            let val: f64 = fld.value().clone().try_into().unwrap();
                            let val_cvt = cvt_distance(val as f32, &user_unit);
                            match user_unit {
                                Units::US => {
                                    let value_str = format!(
                                        "{:<40}: {:<.2} {:<}\n",
                                        fld.name(),
                                        val_cvt,
                                        "miles"
                                    );
                                    text_buffer.insert(&mut end, &value_str);
                                }
                                Units::Metric => {
                                    let value_str = format!(
                                        "{:<40}: {:<.2} {:<}\n",
                                        fld.name(),
                                        val_cvt,
                                        "kilometers"
                                    );
                                    text_buffer.insert(&mut end, &value_str);
                                }
                                Units::None => {
                                    let value_str =
                                        format!("{:<40}: {:<.2} {:<}\n", fld.name(), val_cvt, "");
                                    text_buffer.insert(&mut end, &value_str);
                                }
                            }
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
                            let val_cvt = cvt_temperature(val as f32, &user_unit);
                            match user_unit {
                                Units::US => {
                                    let value_str =
                                        format!("{:<40}: {:<.2} {:<}\n", fld.name(), val_cvt, "°F");
                                    text_buffer.insert(&mut end, &value_str);
                                }
                                Units::Metric => {
                                    let value_str =
                                        format!("{:<40}: {:<.2} {:<}\n", fld.name(), val_cvt, "°C");
                                    text_buffer.insert(&mut end, &value_str);
                                }
                                Units::None => {
                                    let value_str =
                                        format!("{:<40}: {:<.2} {:<}\n", fld.name(), val_cvt, "");
                                    text_buffer.insert(&mut end, &value_str);
                                }
                            }
                        }
                        "enhanced_avg_speed" | "enhanced_max_speed" => {
                            let val: f64 = fld.value().clone().try_into().unwrap();
                            let val_cvt = cvt_pace(val as f32, &user_unit);
                            match user_unit {
                                Units::US => {
                                    let value_str = format!(
                                        "{:<40}: {:<.2} {:<}\n",
                                        fld.name(),
                                        val_cvt,
                                        "min/mile"
                                    );
                                    text_buffer.insert(&mut end, &value_str);
                                }
                                Units::Metric => {
                                    let value_str = format!(
                                        "{:<40}: {:<.2} {:<}\n",
                                        fld.name(),
                                        val_cvt,
                                        "min/km"
                                    );
                                    text_buffer.insert(&mut end, &value_str);
                                }
                                Units::None => {
                                    let value_str =
                                        format!("{:<40}: {:<.2} {:<}\n", fld.name(), val_cvt, "");
                                    text_buffer.insert(&mut end, &value_str);
                                }
                            }
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
        text_buffer.insert(&mut end, "\n");
        for (z, val) in zone_times.iter().enumerate() {
            let val_cvt = cvt_elapsed_time(*val as f32);
            let ll: f64;
            let ul: f64;
            if z == 0 {
                ll = 0.0;
                ul = zone_limits[z];
            } else if z < zone_limits.len() && z > 0 {
                ll = zone_limits[z - 1];
                ul = zone_limits[z];
            } else {
                ll = zone_limits[z - 1];
                ul = 220.0;
            }
            let value_str = format!(
                "{:<5}{:<} ({:>3}-{:>3} bpm): {:01}h:{:02}m:{:02}s\n",
                "Zone", z, ll as i32, ul as i32, val_cvt.0, val_cvt.1, val_cvt.2
            );
            text_buffer.insert(&mut end, &value_str);
        }
        text_buffer.insert(&mut end, "\n");
    };
}

// This is the main body of the program.  After reading the fit file,
// create and display the rest of the UI.
fn parse_and_display_run(
    win: &ApplicationWindow,
    main_pane: &gtk4::Paned,
    data: &Vec<FitDataRecord>,
    units_widget: &DropDown,
) {
    // 1. Instantiate the main UI widgets.
    let text_view = TextView::builder().monospace(true).margin_start(10).build();
    let frame_left = Frame::builder().build();
    let frame_right = Frame::builder().build();
    let left_frame_pane = gtk4::Paned::builder()
        .orientation(Orientation::Vertical)
        .build();
    let right_frame_pane = gtk4::Paned::builder()
        .orientation(Orientation::Horizontal)
        .build();
    let scrolled_window = ScrolledWindow::builder().child(&text_view).build();
    let da_window = ScrolledWindow::builder()
        .vexpand(true)
        .hexpand(true)
        .build();
    let curr_pos_adj = Adjustment::builder()
        .lower(0.0)
        .upper(1.0)
        .step_increment(0.01)
        .page_increment(0.01)
        .value(0.0)
        .build();
    let curr_pos_scale = Scale::builder()
        .orientation(Orientation::Horizontal)
        .adjustment(&curr_pos_adj)
        .draw_value(false)
        .vexpand(false)
        .width_request(120)
        .height_request(30)
        .build();
    let y_zoom_adj = Adjustment::builder()
        .lower(0.5)
        .upper(4.0)
        .step_increment(0.1)
        .page_increment(0.1)
        .value(2.0)
        .build();
    let y_zoom_scale = Scale::builder()
        .orientation(Orientation::Horizontal)
        .adjustment(&y_zoom_adj)
        .draw_value(false)
        .vexpand(false)
        .width_request(120)
        .height_request(30)
        .build();
    let curr_pos_label = Label::new(Some("🏃‍➡️"));
    let y_zoom_label = Label::new(Some("🔍"));
    let controls_box = gtk4::Box::new(Orientation::Vertical, 10);

    // 2. Instantiate embedded widgets based on parsed fit data.
    let (shumate_map, shumate_marker_layer) = build_map(&data);
    let (da, _, yzm, curr_pos) = build_da(&data, &units_widget);
    let text_buffer = text_view.buffer();
    build_summary(&data, &units_widget, &text_buffer);

    // 3. Connect embedded widgets to their parents.
    da_window.set_child(Some(&da));
    frame_right.set_child(Some(&da_window));
    if shumate_map.is_some() {
        frame_left.set_child(shumate_map.as_ref());
    }
    y_zoom_scale.set_adjustment(&yzm);
    curr_pos_scale.set_adjustment(&curr_pos);

    // 4. Configure the widget layout.
    left_frame_pane.set_start_child(Some(&frame_left));
    left_frame_pane.set_end_child(Some(&scrolled_window));
    right_frame_pane.set_start_child(Some(&frame_right));
    controls_box.append(&y_zoom_label);
    controls_box.append(&y_zoom_scale);
    controls_box.append(&curr_pos_label);
    controls_box.append(&curr_pos_scale);
    right_frame_pane.set_end_child(Some(&controls_box));
    // Main box contains all of the above plus the graphs.
    main_pane.set_start_child(Some(&left_frame_pane));
    main_pane.set_end_child(Some(&right_frame_pane));

    // 5. Size the widgets.
    scrolled_window.set_size_request(500, 300);
    // Set where the splits start (in pixels from the left hand side.)
    let main_split = (0.3 * win.width() as f32).trunc() as i32;
    let right_frame_split = (0.7 * (win.width() as f32 - main_split as f32)).trunc() as i32;
    main_pane.set_position(main_split);
    right_frame_pane.set_position(right_frame_split);
    // Set where the splits start (in pixels from the top.)
    left_frame_pane.set_position((0.5 * win.height() as f32) as i32);

    // 6. Configure widgets not handled during instantiation.
    y_zoom_scale.set_draw_value(false); // Ensure the value is not displayed on the scale itself
    curr_pos_scale.set_draw_value(false); // Ensure the value is not displayed on the scale itself

    // 7. Establish call-back routines for widget event handling.
    // Redraw the drawing area when the zoom changes.
    y_zoom_scale.adjustment().connect_value_changed(clone!(
        #[strong]
        da,
        move |_| da.queue_draw()
    ));
    // Redraw the drawing area and map when the current postion changes.
    curr_pos_scale.adjustment().connect_value_changed(clone!(
        #[strong]
        da,
        #[strong]
        data,
        #[strong]
        shumate_map,
        #[strong]
        shumate_marker_layer,
        #[strong]
        curr_pos,
        move |_| {
            // Update graphs.
            da.queue_draw();
            // Update map.
            if shumate_marker_layer.is_some() {
                shumate_marker_layer.as_ref().unwrap().remove_all();
            }
            let units_widget = DropDown::builder().build(); // bogus value - no units required for position
            let run_path = get_xy(&data, &units_widget, "position_lat", "position_long");
            let idx = (curr_pos.value() * (run_path.len() as f64 - 1.0)).trunc() as usize;
            let curr_lat = run_path.clone()[idx].0;
            let curr_lon = run_path.clone()[idx].1;
            let lat_deg = semi_to_degrees(curr_lat);
            let lon_deg = semi_to_degrees(curr_lon);
            let marker_text = Some(get_symbol(&data));
            let marker_content = gtk4::Label::new(marker_text);
            marker_content.set_halign(gtk4::Align::Center);
            marker_content.set_valign(gtk4::Align::Baseline);
            // Style the symbol with mark-up language.
            marker_content.set_markup(Some(get_symbol(&data)).expect("No symbol."));
            let widget = &marker_content;
            let marker = Marker::builder()
                //            .label()
                .latitude(lat_deg)
                .longitude(lon_deg)
                .child(&widget.clone())
                // Set the visual content widget
                .build();
            if shumate_marker_layer.is_some() {
                shumate_marker_layer.as_ref().unwrap().add_marker(&marker);
            }
            if shumate_map.is_some() {
                shumate_map.as_ref().unwrap().queue_draw();
            }
        },
    ));
}

fn build_gui(app: &Application) {
    let win = ApplicationWindow::builder()
        .application(app)
        .title("SiliconSneaker II")
        .build();

    // Main horizontal container to hold the two frames side-by-side,
    // outer box wraps main_pane.
    let outer_box = gtk4::Box::builder()
        .orientation(Orientation::Vertical)
        .spacing(10)
        .build();
    let button_box = gtk4::Box::builder()
        .orientation(Orientation::Horizontal)
        .vexpand(false)
        .hexpand(false)
        .width_request(200)
        .height_request(20)
        .spacing(10)
        .build();
    let main_pane = gtk4::Paned::builder().build();

    // Button with icon and label.
    let button_content = gtk4::Box::new(Orientation::Horizontal, 6);
    button_content.set_halign(gtk4::Align::Center);
    // "document-open" is a standard Freedesktop icon name.
    let icon = Image::from_icon_name("document-open");
    let label = Label::new(Some("Select a FIT file..."));
    button_content.append(&icon);
    button_content.append(&label);
    let btn = Button::builder()
        .child(&button_content)
        .margin_top(5)
        .margin_bottom(5)
        .margin_start(5)
        .margin_end(5)
        .height_request(30)
        .width_request(50)
        .build();
    let uom = StringList::new(&["🇪🇺 Metric", "🇺🇸 US"]);
    let units_widget = DropDown::builder()
        .model(&uom)
        .margin_top(5)
        .margin_bottom(5)
        .margin_start(5)
        .margin_end(5)
        .height_request(30)
        .width_request(100)
        .build();
    let about_label = Label::new(Some("About"));
    let about_btn = Button::builder()
        .child(&about_label)
        .margin_top(5)
        .margin_bottom(5)
        .margin_start(5)
        .margin_end(5)
        .height_request(30)
        .width_request(50)
        .build();

    btn.connect_clicked(clone!(
        #[strong]
        win,
        #[strong]
        main_pane,
        #[strong]
        units_widget,
        move |_| {
            // 1. Create the Native Dialog
            // Notice the arguments: Title, Parent Window, Action, Accept Label, Cancel Label
            let native = FileChooserNative::new(
                Some("Open File Native"),
                Some(&win),
                FileChooserAction::Open,
                Some("Open"),   // Custom label for the "OK" button
                Some("Cancel"), // Custom label for the "Cancel" button
            );

            // 2. Connect to the response signal
            native.connect_response(clone!(
                #[strong]
                win,
                #[strong]
                main_pane,
                #[strong]
                units_widget,
                move |dialog, response| {
                    if response == ResponseType::Accept {
                        // Extract the file path
                        if let Some(file) = dialog.file() {
                            if let Some(path) = file.path() {
                                let path_str = path.to_string_lossy();
                                // Get values from fit file.
                                let file_result = File::open(&*path_str);
                                let mut file = match file_result {
                                    Ok(file) => {
                                        let c_title = win.title().unwrap().to_string().to_owned();
                                        let mut pfx = c_title
                                            .chars()
                                            .take_while(|&ch| ch != ':')
                                            .collect::<String>();
                                        pfx.push_str(":");
                                        pfx.push_str(" ");
                                        pfx.push_str(&path_str);
                                        win.set_title(Some(&pfx.to_string()));
                                        file
                                    }
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
                                if let Ok(data) = fitparser::from_reader(&mut file) {
                                    parse_and_display_run(&win, &main_pane, &data, &units_widget);
                                    // Hook-up the units_widget change handler.
                                    units_widget.connect_selected_notify(clone!(
                                        #[strong]
                                        win,
                                        #[strong]
                                        main_pane,
                                        #[strong]
                                        data,
                                        move |me| {
                                            parse_and_display_run(&win, &main_pane, &data, &me);
                                        }
                                    ));
                                }
                            }
                        }
                    } else {
                        // println!("User cancelled");
                    }
                    // unlike FileChooserDialog, 'native' creates a transient reference.
                    // It's good practice to drop references, but GTK handles the cleanup
                    // once it goes out of scope or the window closes.
                }
            ));
            // 3. Show the dialog
            native.show();
        }
    )); //button-connect-clicked

    about_btn.connect_clicked(clone!(
        #[strong]
        win,
        move |_| {
            let dialog = gtk4::AboutDialog::builder()
                .transient_for(&win)
                .modal(true)
                .program_name("SiliconSneaker II")
                .version("1.0.0")
                .copyright("Copyright © 2025")
                .comments("View your run files on the desktop!")
                .authors(vec![
                    "Craig S. Prevallet <penguintx@hotmail.com>".to_string(),
                ])
                .build();
            dialog.present();
        }
    )); //button-connect-clicked

    button_box.append(&btn);
    button_box.append(&units_widget);
    button_box.append(&about_btn);
    outer_box.append(&button_box);
    outer_box.append(&main_pane);
    win.set_child(Some(&outer_box));
    win.maximize();
    win.present();
} // build_gui
