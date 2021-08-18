eps = 0.05; // tolerance value 
screen_dims = [166, 106, 12]; // x,y,z dimensions of screen in milimeters, depends on screen
screw_port_dims = [8.5, 9, screen_dims.z]; // x,y,z dimensions of screen screw holes relative to the corner of the screen, depends on screen
board_dims = [];
board_pos1 = [-63,45,5];
board_pos2 = [63,-45,5];

/**
Note that base_height needs to be greater than height of Jetson and height of battery pack(s).
**/
base_height = 45; // Height of base of top housing component measured from bottom
bottom_height = 25; // Height of bottom housing component

// LCD casing is rotated to give a 10 degree slant
LCD_slant = 10;

// Positions of LCD ports to hollow out later
module lcd_screw_ports(screen = screen_dims, screw = screw_port_dims) {
    translate([screen.x/2-screw.x, screen.y/2, 0]) cube(screw_port_dims);
    mirror([1,0,0]) translate([screen.x/2-screw.x, screen.y/2, 0]) cube(screw_port_dims);
}

// Basic structure of screen to hollow out space later
module screen(screen = screen_dims) {
    translate([-screen.x/2, -screen.y/2, 0]) cube(screen);
}

// Make a 2D footprint of the basic LCD screen structure used to offset and extrude later to form outer structure of LCD housing
module lcd_footprint(screen = screen_dims, screw = screw_port_dims) {
    offset(2) projection() union() {
        lcd_screw_ports();
        mirror([0,1,0]) lcd_screw_ports();
        translate([-screen.x/2, -screen.y/2, 0]) cube(screen);
    }
}

// Space for LCD body, depends on screen
module lcd_hollow(screen = screen_dims, screw = screw_port_dims) {
    linear_extrude(height = screen_dims.z) lcd_footprint();
}

// Space for wiring to LCD, depends on screen
module lcd_wiring_hollow(screen = screen_dims, screw = screw_port_dims) {
    lcd_wiring_space = [20+3,65,10]; // Dimension of space to be hollowed, depends on screen and wires
    wiring_z_offset = 17; // Z offset for wiring to give extra space
    // Horizontal opening on the right side for wiring
    translate([screen.x/2-lcd_wiring_space.x+5+eps,-screw.y,screen.z-wiring_z_offset]) cube(lcd_wiring_space);
    // Extra space for bulkier HDMI cord
    translate([screen.x/2+eps,screen.y/2-2*screw.y,screen.z-wiring_z_offset]) cube([5,20,wiring_z_offset]);
}

// Outer structure of LCD housing
module lcd_casing_structure() {
    linear_extrude(height = screen_dims.z + 7) offset(3) lcd_footprint();
}

// Screw holes to secure LCD
module lcd_screw_holes(screen = screen_dims, screw = screw_port_dims) {
    // Holes are defined as with the cylinder function with $fn as 3 so that they are essentially triangular
    lcd_hole_d = 4; // Diameter of LCD screw hole, depends on size of screw used 
    module screw_hole(screen = screen_dims, screw = screw_port_dims) {
        translate([screen_dims.x/2-screw.x/2,screen_dims.y/2+screw.y/2,2]) cylinder(999, d = lcd_hole_d, $fn = 3);
        mirror([1,0,0]) translate([screen_dims.x/2-screw.x/2,screen_dims.y/2+screw.y/2,2]) cylinder(999, d = lcd_hole_d, $fn = 3);
    }
    screw_hole();
    mirror([0,1,0]) screw_hole();
}

// LCD housing to be rotated
module lcd_casing_top() {
    difference() {
        // Outer structure shell for LCD housing
        lcd_casing_structure();
        // Hollow out space for LCD, wiring, and screw ports 
        union() {
            translate([0,0,7+eps]) lcd_hollow();
            translate([0,0,7+eps]) lcd_wiring_hollow();
            lcd_screw_holes();
        }
    }
}

// Rotated LCD housing
module lcd_casing(screen = screen_dims, screw = screw_port_dims) {
    // Rotated to give a slant for ease of usage 
    translate([0,0,(screen.y+2*screw.y+10)*sin(LCD_slant)/2]) rotate(LCD_slant, [1,0,0]) lcd_casing_top();
}

// Two legs to support slanted LCD housing
module lcd_support(screen = screen_dims, screw = screw_port_dims, column_pos = 60) {
    // Make one of the legs
    module lcd_support_column(screen = screen, screw = screw) {
        /**
        The supports are essentially two cylinders hulled together, so support_dims_back and support_dims_front gives the dimensions of the taller and shorter cylinders respectively
        Note these are defined relative to screen and screw ports (this can be changed)
        **/
        support_dims_back = [10,5,(screen.y+2*screw.y+10)*tan(LCD_slant)+1];
        support_dims_front = [10,5,(screen.y+2*screw.y+10)*tan(LCD_slant)/3*2+1];
        // Hull the cylinders together to form the column
        hull() {
            translate([0,(screen.y+2*screw.y+10)*cos(LCD_slant)/2*0.25,-2]) rotate(LCD_slant,[1,0,0]) cylinder(h = support_dims_front.z, r = support_dims_front.y);
            translate([0,(screen.y+2*screw.y+10)*cos(LCD_slant)/2*0.75,-2]) rotate(LCD_slant,[1,0,0]) cylinder(h = support_dims_back.z, r = support_dims_back.y);
        }
    }
    // Make both legs
    leg_pos = 60; // Position of legs
    translate([leg_pos,0,0]) lcd_support_column();
    translate([-leg_pos,0,0]) lcd_support_column();
}

// LCD housing and support 
module lcd_full(screen = screen_dims, screw = screw_port_dims) {
    lcd_casing();
    lcd_support(screen = screen_dims, screw = screw_port_dims);
}

// Model of Jetson Nano used to hollow out space later, quick and easy way to get dimensions and hole positions (could also use cube() or an STL of a different board if needed)
module jetson_space() {
    // Credit to https://grabcad.com/library/jetson-nano-3
    import("./libs/jetson_nano.stl");
}

// The battery pack used in the default setup was originally built as a HAT for Jetosn Nano, so it has the same dimensions and screw hole positions, thus we just repeat the same dimensions and positions twice
module boards(board_pos1, board_pos2) {
    hull() {
        // Change one of these to desired dimensions for custom battery pack(s), leave the other one for Jetson Nano
        translate(board_pos1) rotate(-90, [0,0,1]) jetson_space();
        translate(board_pos2) rotate(90, [0,0,1]) jetson_space();
    }
}

// Hollow out space for board wiring and cords
module board_wiring_cutouts() {
    translate([0,0,15]) cube([999,100,23], center = true);
}

// Hollow out holes for bolt
module hole(screen = screen_dims, screw = screw_port_dims, hole_z, hole_d) {
    translate([screen.x/2-screw.x/2, screen.y/2+screw.y/2+5,hole_z+eps]) cylinder(22, d = hole_d, $fn=25);
}
    
module bottom_base_screw(screen = screen_dims, screw = screw_port_dims) {
    // Define rectangular slot for ease of access to secure bolt and nut
    module slot(screen = screen, screw = screw) { 
        slot_dims = [6,10,base_height-bottom_height];
        translate([screen.x/2-screw.x+slot_dims.x/4,screen.y/2+screw.y/2-slot_dims.y/4+4,-eps]) cube(slot_dims);
    }
    bottom_base_hole_d = 4; // Diameter of hole for bolt
    bottom_base_hole_z = 5; // Height of hole for bolt to hollow out
    // Form slot and holes and reflect to make all four
    union() {
        hole(hole_z = bottom_base_hole_z, hole_d = bottom_base_hole_d);
        slot();
        mirror([1,0,0]) slot();
        mirror([1,0,0]) hole(hole_z = bottom_base_hole_z, hole_d = bottom_base_hole_d);
    }
    mirror([0,1,0]) union() {
        hole(hole_z = bottom_base_hole_z, hole_d = bottom_base_hole_d);
        slot();
        mirror([1,0,0]) slot();
        mirror([1,0,0]) hole(hole_z = bottom_base_hole_z, hole_d = bottom_base_hole_d);
    }
}

// Screw holes to connect top_base and bottom_base together
module top_base_screw(screen = screen_dims, screw = screw_port_dims) {
    top_base_hole_d = 4; // Diameter of hole for bolt
    top_base_hole_z = 15; // Height of hole for bolt to hollow out
    union() {
        hole(hole_z = top_base_hole_z, hole_d = top_base_hole_d);
        mirror([1,0,0]) hole(hole_z = top_base_hole_z, hole_d = top_base_hole_d);
    }
    mirror([0,1,0]) union() {
        hole(hole_z = top_base_hole_z, hole_d = top_base_hole_d);
        mirror([1,0,0]) hole(hole_z = top_base_hole_z, hole_d = top_base_hole_d);
    }
    /**
    Hollow out space for nut, use same dimensions as the space for the bottom base.
    This can be concealed better to improve aesthetics but we chose to keep it exposed for flexibility of bolt lengths and nut sizes
    **/
    translate([0,0,29]) linear_extrude(16+eps) projection() bottom_base_screw();
}

// Screw supports and holes for Jetson Nano and battery pack(s)
module bottom_base_board_screw() {
    // Location of screw holes by Jetson Nano specification
    top_left = [-63.,45,2-eps];
    top_right = [-5,45,2-eps];
    bottom_left = [-63,-41,2-eps];
    bottom_right = [-5,-41,2-eps];
    // Extruded cylindrical support to hold up Jetson and battery pack, can be changed depending on battery pack(s) used
    module supports(support_h = 3, support_d = 5) {
        $fn = 25;
        translate(top_left) cylinder(support_h+eps, d = support_d); 
        translate(bottom_left) cylinder(support_h+eps, d = support_d); 
        translate(top_right) cylinder(support_h+eps, d = support_d); 
        translate(bottom_right) cylinder(support_h+eps, d = support_d); 
    }
    /**
    Holes for screw to secure Jeston and battery, can be changed depending on battery pack(s) used. 
    Holes are essentially cylinders with $fn as 4 to make them rectangular. 
    **/
    module holes(hole_h = 10, hole_d = 2.5*1.25) {
        $fn = 4;
        translate(top_left) cylinder(hole_h+eps, d = hole_d); 
        translate(bottom_left) cylinder(hole_h+eps, d = hole_d); 
        translate(top_right) cylinder(hole_h+eps, d = hole_d); 
        translate(bottom_right) cylinder(hole_h+eps, d = hole_d); 
    }
    difference() {
        supports();
        holes();
    }
    // Since our battery pack was originally a HAT for the Jetson Nano, they have the same screw port dimensions, so we can simply reflect 
    translate([0,-top_left.y-bottom_left.y,0]) mirror([1,0,0]) difference() { 
        supports();
        holes();
    } // Translated so that battery pack faces the desired direction
}

// Generate top base that houses LCD screen
module top_base(base_height, bottom_height, board_pos1, board_pos2) {
    // Make upper half of top_base (LCD housing)
    translate([0,0,base_height-2]) lcd_full();
    // Make lower half of top_base
    difference() {
        // Lower half of top_base
        linear_extrude(base_height) offset(3) lcd_footprint();
        // Hollow out space for bottom_base and Jetson and battery
        translate([0,0,-eps]) {
            linear_extrude(bottom_height+eps) offset(3+eps) lcd_footprint();
            linear_extrude(base_height+2*eps) projection() translate([0,-bottom_height+2,0]) cube([162, base_height, 12], center = true);
            linear_extrude(base_height-5+eps) offset(3) projection() boards(board_pos1, board_pos2);
        }
        top_base_screw();
    }
}

// Generate bottom base that houses Jetson Nano and battery pack(s)
module bottom_base(bottom_height, board_pos1, board_pos2) {
    difference() {
        linear_extrude(bottom_height) offset(3) lcd_footprint();
        translate([0,0,2]) linear_extrude(bottom_height) offset(3) projection() boards(board_pos1, board_pos2);
        board_wiring_cutouts();
        bottom_base_screw();
    }
    bottom_base_board_screw();
}

top_base(base_height, bottom_height, board_pos1, board_pos2);
bottom_base(bottom_height, board_pos1, board_pos2);