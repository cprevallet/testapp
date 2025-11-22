use gtk4::prelude::*;
use gtk4::{Application, ApplicationWindow, DrawingArea, Frame, Orientation};
use plotters::prelude::*;
//use shumate::prelude::*;
//use gtk4::glib::clone;
use fitparser::{FitDataRecord, profile::field_types::MesgNum};
use shumate::Map;
use std::fs::File;

// Only God and I knew what this was doing when I wrote it.
// Know only God knows.

// Global, compile-time constant strings
const FIT_FILE_NAME: &'static str = "tests/broken.fit";
// X_PARAM and Y_PARAM can have the value of:
// distance
// enhanced_altitude
// enhanced_speed
// heart_rate
// cadence
// position_lat
// position_long
const XPARAM: &'static str = "distance";
const YPARAM: &'static str = "enhanced_speed";

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
fn get_plot_range(data: Vec<(f32, f32)>) -> (std::ops::Range<f32>, std::ops::Range<f32>) {
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
    let yrange: std::ops::Range<f32> = mean_y - 3.0 * sigma_y..mean_y + 3.0 * sigma_y;
    return (xrange, yrange);
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

// Convert speed (m/s) to pace(min/mile)
fn cvt_to_pace(speed: f32) -> f32 {
    if speed < 1.00 {
        return 26.8224; //avoid divide by zero
    } else {
        return 26.8224 / speed;
    }
}

// Retrieve raw values to plot from fit file.
fn get_xy(data: Vec<FitDataRecord>, x_field_name: &str, y_field_name: &str) -> Vec<(f32, f32)> {
    let mut xy_pairs: Vec<(f32, f32)> = Vec::new();
    // Parameter can be distance, heart_rate, enhanced_speed, enhanced_altitude.
    let x: Vec<f64> = get_msg_record_field_as_vec(data.clone(), x_field_name);
    let y: Vec<f64> = get_msg_record_field_as_vec(data.clone(), y_field_name);
    //  Convert values to 32 bit and create a tuple.
    if (x.len() == y.len()) && (x.len() != 0) && (y.len() != 0) {
        for index in 0..x.len() - 1 {
            //TODO This is ugly!  Need a better method to handle conversions.
            if y_field_name != "enhanced_speed" {
                xy_pairs.push((x[index] as f32, y[index] as f32));
            } else {
                // special case
                let pace = cvt_to_pace(y[index] as f32);
                xy_pairs.push((x[index] as f32, pace as f32));
            }
        }
    };
    return xy_pairs;
}

// Build drawing area.
fn build_da() -> DrawingArea {
    let drawing_area: DrawingArea = DrawingArea::builder().build();
    // Get values from fit file.
    let mut plotvals: Vec<(f32, f32)> = Vec::new();
    //println!("Parsing FIT files using Profile version: {:?}", fitparser::profile::VERSION);
    let mut fp = File::open(FIT_FILE_NAME).expect("file not found");
    if let Ok(data) = fitparser::from_reader(&mut fp) {
        plotvals = get_xy(data, XPARAM, YPARAM);
    }
    //  Find the plot range (minx..maxx, miny..maxy)
    let plot_range = get_plot_range(plotvals.clone());
    // Format the labels on the y-axis.
    let num_formatter = |x: &f32| format!("{:.3}", x);
    let pace_formatter = |x: &f32| {
        let mins = x.trunc();
        let secs = x - mins;
        format!("{:02.0}:{:02.0}", mins, secs)
    };
    // Wrap up the data structure to use in draw_func.
    struct PlotData<'a> {
        plotvals: Vec<(f32, f32)>,
        caption: &'a str,
        xlabel: &'a str,
        ylabel: &'a str,
        plot_range: (std::ops::Range<f32>, std::ops::Range<f32>),
        y_formatter: Box<dyn Fn(&f32) -> String>,
    }
    let mut pd = PlotData {
        plotvals: plotvals,
        caption: "",
        xlabel: "Distance",
        ylabel: "",
        plot_range: plot_range,
        y_formatter: Box::new(num_formatter),
    };
    if YPARAM == "enhanced_altitude" {
        pd.caption = "Elevation";
        pd.ylabel = "Elevation";
    }
    if YPARAM == "cadence" {
        pd.caption = "Cadence";
        pd.ylabel = "Cadence";
    }
    if YPARAM == "heart_rate" {
        pd.caption = "Heart Rate";
        pd.ylabel = "Heart Rate";
    }
    // Special handling for pace plots.
    if YPARAM == "enhanced_speed" {
        pd.caption = "Pace";
        pd.ylabel = "Pace(min/mile)";
        pd.y_formatter = Box::new(pace_formatter);
    }

    // Use a "closure" (anonymous function?) as the drawing area draw_func.
    // The pd struct is passed in.
    drawing_area.set_draw_func(move |_drawing_area, cr, width, height| {
        // --- 🎨 Custom Drawing Logic Starts Here ---
        let root = plotters_cairo::CairoBackend::new(
            &cr,
            (width.try_into().unwrap(), height.try_into().unwrap()),
        )
        .unwrap()
        .into_drawing_area();
        let _ = root.fill(&WHITE);
        let root = root.margin(50, 50, 50, 50);
        // After this point, we should be able to construct a chart context
        //
        let mut chart = ChartBuilder::on(&root)
            // Set the caption of the chart
            .caption(pd.caption, ("sans-serif", 40).into_font())
            // Set the size of the label region
            .x_label_area_size(100)
            .y_label_area_size(100)
            // Finally attach a coordinate on the drawing area and make a chart context
            .build_cartesian_2d(pd.plot_range.clone().0, pd.plot_range.clone().1)
            .unwrap();
        let _ = chart
            .configure_mesh()
            // We can customize the maximum number of labels allowed for each axis
            .x_labels(15)
            .y_labels(5)
            .x_desc(pd.xlabel)
            .y_desc(pd.ylabel)
            .y_label_formatter(&pd.y_formatter)
            .draw();
        // // And we can draw something in the drawing area
        // We need to clone plotvals each time we make a call to LineSeries and PointSeries
        let _ = chart.draw_series(LineSeries::new(pd.plotvals.clone(), &RED));
        let _ = root.present();
        // --- Custom Drawing Logic Ends Here ---
    });
    return drawing_area;
}

// Build the map.
fn build_map() -> Map {
    return Map::new_simple();
}

// Create the GUI.
fn build_gui(app: &Application) {
    let win = ApplicationWindow::builder()
        .application(app)
        .default_width(1024)
        .default_height(768)
        .title("Test")
        .build();
    let da = build_da();
    let shumate_map = build_map();
    // Frame 1: Controls
    let frame_left = Frame::builder()
        .label("Frame 1: Controls")
        .child(&shumate_map)
        //        .margin_all(5)
        .build();
    // Frame 2: Drawing Output
    let frame_right = Frame::builder()
        .label("Frame 2: Drawing Area")
        .child(&da)
        //        .margin_all(5)
        .build();
    // Main horizontal container to hold the two frames side-by-side
    let main_box = gtk4::Box::new(Orientation::Horizontal, 10);
    main_box.set_homogeneous(true); // Ensures both frames take exactly half the window width
    main_box.append(&frame_left);
    main_box.append(&frame_right);
    win.set_child(Some(&main_box));
    //win.set_child(Some(&da));
    // win.set_child(Some(&shumate_map));
    //    win.set_child(Some(&shumate_map));
    win.present();
}
