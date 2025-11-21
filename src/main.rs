use gtk4::glib::closure::IntoClosureReturnValue;
use gtk4::prelude::*;
use plotters::prelude::*;
use gtk4::{Application, ApplicationWindow, DrawingArea};
use gtk4::glib::clone;
use std::fs::File;
use fitparser::{profile::field_types::MesgNum, FitDataField, FitDataRecord};

// Only God and I knew what this was doing when I wrote it.
// Know only God knows.

fn main() {
    let app = Application::builder().build();
    app.connect_activate(build_gui);
    app.run();
}

// Find the largest non-NaN in vector, or NaN otherwise:
fn max_vec(vector : Vec<f32>) -> f32  {
    let v = vector.iter().cloned().fold(0./0., f32::max);
    return v;
}

// Find the largest non-NaN in vector, or NaN otherwise:
fn min_vec(vector : Vec<f32>) -> f32  {
    let v = vector.iter().cloned().fold(0./0., f32::min);
    return v;
}

// Split vector of tuples into two vecs
fn get_plot_range(data : Vec<(f32, f32)>) -> (std::ops::Range<f32>, std::ops::Range<f32>) {
    let (x, y): (Vec<_>, Vec<_>) = data.into_iter().map(|(a, b)| (a, b)).unzip();    
    // Find the range of the chart
    let xrange : std::ops::Range<f32> = min_vec(x.clone())..max_vec(x.clone());
    let yrange : std::ops::Range<f32> = min_vec(y.clone())..max_vec(y.clone());
    return (xrange, yrange);
}

// Return a vector of values of "field_name".
fn get_msg_record_field_as_vec(data : Vec<FitDataRecord>, field_name :  &str) -> Vec<f64> {
    let mut field_vals : Vec<f64> = Vec::new();
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
            },
            _ => (), // matches other patterns 
        }           
    }
    return field_vals;
}

// Retrieve values to plot from fit file.
fn get_xy(data : Vec<FitDataRecord>, x_field_name :  &str, y_field_name : &str) ->  Vec<(f32, f32)>{
    let mut xy_pairs: Vec<(f32, f32)> = Vec::new();
    // Parameter can be distance, heart_rate, enhanced_speed, enhanced_altitude.
    let x : Vec<f64> = get_msg_record_field_as_vec(data.clone(), x_field_name);
    let y : Vec<f64> = get_msg_record_field_as_vec(data.clone(), y_field_name);
    // Create vector of tuples from individual vectors.
//  Convert values to 32 bit and create a tuple.
    if x.len() == y.len() {
        for index in 0..x.len()-1 {
            xy_pairs.push((x[index] as f32, y[index] as f32));
        }
    };
    return xy_pairs;
}

fn build_gui(app: &Application){
    let win = ApplicationWindow::builder().application(app).default_width(1024).default_height(768).title("Test").build();
    let drawing_area: DrawingArea = DrawingArea::builder().build();

    // Get values from fit file.
    let mut plotvals: Vec<(f32, f32)> = Vec::new();
    println!("Parsing FIT files using Profile version: {:?}", fitparser::profile::VERSION);
    let mut fp = File::open("tests/working.fit").expect("file not found");
    if let Ok(data) = fitparser::from_reader(&mut fp) {
        // plotvals = get_xy(data, "distance", "enhanced_speed");
        // plotvals = get_xy(data, "distance", "enhanced_altitude");
        // plotvals = get_xy(data, "distance", "heart_rate");
        plotvals = get_xy(data, "distance", "cadence");
        // plotvals = get_xy(data, "distance", "position_lat");
        // plotvals = get_xy(data, "distance", "position_long");
    }

    // Use a "closure" (anonymous function?) as the drawing area draw_func.
    // We pass a strong reference to the plot data (aka plotvals).
    drawing_area.set_draw_func(clone!(#[strong] plotvals, move |_drawing_area, cr, width, height| {
        // --- 🎨 Custom Drawing Logic Starts Here ---
 
        let root = plotters_cairo::CairoBackend::new(&cr, (width.try_into().unwrap(), height.try_into().unwrap())).unwrap().into_drawing_area();
        let _ = root.fill(&WHITE);

        let root = root.margin(50, 50, 50, 50);

        //  Find the plot range (minx..maxx, miny..maxy)
        let plot_range = get_plot_range(plotvals.clone());
        println!("{:?}", plot_range);
        
        // After this point, we should be able to construct a chart context
        //
       
        let mut chart = ChartBuilder::on(&root)
            // Set the caption of the chart
            .caption("This is our first plot", ("sans-serif", 40).into_font())
            // Set the size of the label region
            .x_label_area_size(20)
            .y_label_area_size(40)
            // Finally attach a coordinate on the drawing area and make a chart context
            // .build_cartesian_2d(plot_range.0, plot_range.1).unwrap();
            .build_cartesian_2d(plot_range.0, plot_range.1).unwrap();

        // Then we can draw a mesh
        let _ = chart
            .configure_mesh()
            // We can customize the maximum number of labels allowed for each axis
            .x_labels(15)
            .y_labels(5)
            // We can also change the format of the label text
            .y_label_formatter(&|x| format!("{:.3}", x))
            .draw();

        // And we can draw something in the drawing area
        // We need to clone plotvals each time we make a call to LineSeries and PointSeries
        let _ = chart.draw_series(LineSeries::new(
              plotvals.clone(),
            &RED,
        ));
        // Similarly, we can draw point series
        // let _ = chart.draw_series(PointSeries::of_element(
        //       plotvals.clone(),
        //     5,
        //     &RED,
        //     &|c, s, st| {
        //         return EmptyElement::at(c)    // We want to construct a composed element on-the-fly
        //         + Circle::new((0,0),s,st.filled()) // At this point, the new pixel coordinate is established
        //         + Text::new(format!("{:?}", c), (10, 0), ("sans-serif", 10).into_font());
        //     },
        // ));
        
        let _ = root.present();
        // --- Custom Drawing Logic Ends Here ---
    }));

    win.set_child(Some(&drawing_area));
    win.present();
}
