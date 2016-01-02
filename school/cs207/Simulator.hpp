#ifndef CS207_SIMULATOR_HPP
#define CS207_SIMULATOR_HPP

#include "CS207/Util.hpp"
#include "Mesh.hpp"
#include "Color.hpp"
#include "Action.hpp"
#include "Point.hpp"
#include <fstream>
#include <cstdio>


#include "CS207/GLCamera.hpp"
#include <SDL/SDL.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <string>
#include <assert.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif
#include "GLTools.hh"

// A default color functor that returns white for anything it recieves

struct DefaultColor {

  template <typename NODE >
    Color operator()(const NODE & node) {
    (void) node;
    return Color(1);
  }
};

// A default select color; white

static const Color default_select_color(1, 1, 1);

// A default functor that evaluates to 'false' as a boolean expression
// Useful for turning certain features off

struct FalseFunctor {

  template <typename NODE >
    Point operator()(const NODE & n) {
    (void) n;
    return Point();
  }

  operator const void*() const {
    return 0;
  }
};

typedef FalseFunctor NoVector;
// A normal vector field functor that returns a node's normal

struct NormalVector {

  template <typename NODE >
    Point operator()(const NODE & n) {
    return n.normal() * .1;
  }

  operator const void*() const {
    return this;
  }
};


typedef FalseFunctor NoNormal;

struct DefaultNormal {

  template <typename NODE >
    Point operator()(const NODE & n) {
    return n.normal();
  }

  operator const void*() const {
    return this;
  }
};

static constexpr int dimension = 3;

class Visualization {
  // Coordinate storage type
  typedef Point::point_type coordinate_type;
  // Vertexes
  std::vector<Point> coords_;
  // Normals
  std::vector<Point> normals_;
  // Vector coordinates
  std::vector<Point> vector_coords_;
  // Color storage type
  typedef Color::color_type color_type;
  // Colors
  std::vector<Color> colors_;
  // Vector colors
  std::vector<Color> vector_colors_;
  /** Type of node indexes. */
  typedef unsigned node_index_type;
  // Edges (index pairs)
  std::vector<node_index_type> edges_;
  // Edges (index pairs)
  std::vector<node_index_type> triangles_;
  // Vector Rendering edges, same format
  std::vector<node_index_type> vector_edges_;
  //vector of the selected nodes
  std::vector<node_index_type> selected_nodes_idxs;

  void clear() {
    coords_.clear();
    normals_.clear();
    colors_.clear();
    edges_.clear();
    triangles_.clear();
    vector_coords_.clear();
    vector_colors_.clear();
    vector_edges_.clear();
  }

  Point center() {

    // Sum each coordinate component
    int n = coords_.size();
    Point center;
    for (int k = 0; k < n; ++k) {
      center += coords_[k];
    }
    center /= n;
    return center;
  }

  template <typename IT, typename C = DefaultColor, typename V = FalseFunctor, typename N = DefaultNormal>
    void add_nodes(IT first, IT last,
    N normal_function = N(), C color_function = C(), V vector_function = V()){

    for (; first != last; ++first) {

      // Get node and record the index mapping
      typename IT::value_type node = *first;
      node_index_type this_index = (*first).index();

      assert(!(this_index > coords_.size()));
      if (this_index == coords_.size()) {
        coords_.resize(coords_.size() + 1);
        colors_.resize(colors_.size() + 1);
        if (normal_function)
          normals_.resize(normals_.size() + 1);

        //four points per node
        vector_coords_.resize(vector_coords_.size() + 4);
        vector_colors_.resize(vector_colors_.size() + 4);

        //draw the edges for the vectors
        //node at index has to be connected to start_vector + index*4, index*4 + 1, index*4 +2, index*4 +3
        //back and front
        vector_edges_.push_back(this_index * 4 + 0);
        vector_edges_.push_back(this_index * 4 + 1);

        //front and left arrow
        vector_edges_.push_back(this_index * 4 + 1);
        vector_edges_.push_back(this_index * 4 + 2);

        //front and right arrow
        vector_edges_.push_back(this_index * 4 + 1);
        vector_edges_.push_back(this_index * 4 + 3);
      }

      // Get node position
      Point p = node.position();
      coords_[this_index] = p;

      // Get node normal
      if (normal_function)
        normals_[this_index] = normal_function(node);

      // Get node color
      Color c = color_function(node);
      colors_[this_index] = c;

      if (vector_function) {
        Point direct = vector_function(node);

        Point behind = direct / -2;
        Point ahead = behind + direct;

        //first point behind
        vector_coords_[this_index * 4] = p + behind;
        //point in front
        vector_coords_[this_index * 4 + 1] = p + ahead;

        //arrows are 1/8 of the way from the first point and .1 from the base with random direction
        Point arrowBase = p + (ahead * 3) / 4;
        // let z = 0 so we only have to solve for x and y
        // a*sample.x + b* sample.y = 0 => a= (-b *sample.y/sample.x) => le b = 1
        Point orthog;
        if (direct.x == 0)
          orthog = Point(1, 1, 0);
        else
          orthog = Point((-1)*(direct.y / direct.x), 1, 0);
        orthog = orthog.normalize();
        orthog *= (ahead / 4).length();

        //arrow left
        vector_coords_[this_index * 4 + 2] = arrowBase + orthog;
        //arrow right
        vector_coords_[this_index * 4 + 3] = arrowBase - orthog;
        //set all the colors of the vector to the initial color
        for (int i = 0; i < 4; i++) {
          vector_colors_[this_index * 4 + i] = c;
        }
      }
    }
  }

  template <typename IT>
  void add_edges(IT first, IT last) {
    edges_.clear();
    for (; first != last; ++first) {
      typename IT::value_type edge = *first;
      node_index_type idx1 = edge.node1().index();
      node_index_type idx2 = edge.node2().index();
      if (idx1 < coords_.size() && idx2 < coords_.size()) {
        edges_.push_back(idx1);
        edges_.push_back(idx2);
      }
    }
  }

  template <typename IT>
  void add_triangles(IT first, IT last) {
    triangles_.clear();
    for (; first != last; ++first) {
      typename IT::value_type triangle = *first;
      node_index_type idx0 = triangle.node(0).index();
      node_index_type idx1 = triangle.node(1).index();
      node_index_type idx2 = triangle.node(2).index();
      if (idx0 < coords_.size() && idx1 < coords_.size() && idx2 < coords_.size()) {
        triangles_.push_back(idx0);
        triangles_.push_back(idx1);
        triangles_.push_back(idx2);
      }
    }
  }

  /** Add to the list of selected nodes n
   *  @param[in]  The index of the node being added to the list of selected nodes in the simulator
   **/
  void select_node(node_index_type idx) {
    if (std::find(selected_nodes_idxs.begin(), selected_nodes_idxs.end(), idx) == selected_nodes_idxs.end()) {
      selected_nodes_idxs.push_back(idx);
    }
  }

  /** Removes a node from the list of selected nodes
   *  @param[in]  The index of the node being removed from the list of selected nodes in the simulator
   **/
  void deselect_node(node_index_type idx) {
    typename std::vector<node_index_type>::iterator sit = std::find(selected_nodes_idxs.begin(), selected_nodes_idxs.end(), idx);
    if (sit != selected_nodes_idxs.end()) {
      selected_nodes_idxs.erase(sit);
    }

  }

  void render(bool vector_mode, bool draw_mode, bool triangle_mode) const {

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    if (vector_mode) {
      glVertexPointer(dimension, gltype<coordinate_type>::value, 0, &vector_coords_[0]);
      glColorPointer(dimension, gltype<color_type>::value, 0, &vector_colors_[0]);
      //      glDrawArrays(GL_POINTS, 0, vector_coords_.size() / dimension);
      glDrawElements(GL_LINES, vector_edges_.size(), gltype<node_index_type>::value, &vector_edges_[0]);
    }

    if (draw_mode) {
      // Load all nodes and colors
      glVertexPointer(dimension, gltype<coordinate_type>::value, 0, &coords_[0]);
      glColorPointer(dimension, gltype<color_type>::value, 0, &colors_[0]);
      //Load the normals as well
      if (triangle_mode && triangles_.size()) {
        //turn on light
        if (normals_.size()) {
          glEnable(GL_LIGHTING);
          glEnableClientState(GL_NORMAL_ARRAY);
          glNormalPointer(gltype<coordinate_type>::value, 0, &normals_[0]);
        }
        // Draw surface
        glDrawElements(GL_TRIANGLES, triangles_.size(), gltype<node_index_type>::value, &triangles_[0]);
        //turn off light
        glDisableClientState(GL_NORMAL_ARRAY);
        glDisable(GL_LIGHTING);
      } else {
        // Draw wireframe mesh
        glPointSize(10);
        glLineWidth(1);
        glDrawArrays(GL_POINTS, 0, coords_.size());
        glDrawElements(GL_LINES, edges_.size(), gltype<node_index_type>::value, &edges_[0]);
      }
      // Load selected nodes and colors
      glDisableClientState(GL_COLOR_ARRAY);
      glColor3f(default_select_color.r, default_select_color.g, default_select_color.b);


      //Draw selected nodes
      glPointSize(20);
      glDrawElements(GL_POINTS, selected_nodes_idxs.size(), gltype<node_index_type>::value, &selected_nodes_idxs[0]);

    }

    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
  }

  friend class Simulator;
};

class Simulator {
private:

  bool quiet_;
  // Rendering surface
  SDL_Surface* surface_;
  // Window width and height
  int window_width_;
  int window_height_;

  // Event handler for interactivity
  SDL_Thread* event_thread_;
  // Synchronization lock
  SDL_mutex* lock_;
  // Queue lock
  bool render_requested_;

  // OpenGL Camera to track the current view
  CS207::GLCamera camera_;

  // Currently displayed label
  std::string label_;

  struct safe_lock {
    Simulator* v_;
    bool ok_;

    safe_lock(Simulator * v)
      : v_(v) {
      if (SDL_LockMutex(v_->lock_)) {
        fprintf(stderr, "\nSDL Mutex lock error: %s", SDL_GetError());
        ok_ = false;
      } else
        ok_ = true;
    }

    ~safe_lock() {
      if (ok_ && SDL_UnlockMutex(v_->lock_))
        fprintf(stderr, "\nSDL Mutex unlock error: %s", SDL_GetError());
    }
  };
  friend struct safe_lock;



public:

  /** Constructor for Simulator
   *  @param[in] p Process to be run before and after a simulation
   *  @pre p has two functions, init which takes a simulator and
   *    operator which takes the current time, dt, and simulator
   *  @pre all nodes added to P are of the same mesh type
   *  @return simulator which visually simulates @a p
   **/
  Simulator(bool quiet = false)
    : quiet_(quiet), surface_(NULL), event_thread_(NULL), lock_(NULL),
    render_requested_(false),
    pause_mode_(false), vector_mode_(false), select_mode_(false), draw_mode_(true), triangle_mode_(false),
    last_x_pos_(0), last_y_pos_(0), select_x_pos_(0), select_y_pos_(0), left_down_(false), right_down_(false),
    vis_(), cur_vis(0), actions_() {
  }

  /** Destructor - Waits until the event thread exits, then cleans up
   *  @pre Simulator has been constructed with the constructor
   */
  ~Simulator() {
    if (event_thread_ != NULL)
      SDL_WaitThread(event_thread_, NULL);
    SDL_FreeSurface(surface_);
    SDL_DestroyMutex(lock_);
    SDL_Quit();
  }

  /** Launch a new SDL window. */
  void launch() {
    if (event_thread_ == NULL) {
      lock_ = SDL_CreateMutex();
      if (lock_ == NULL)
        fprintf(stderr, "Unable to create mutex: %s\n", SDL_GetError());
      if (!quiet_) {
        event_thread_ = SDL_CreateThread(Simulator::event_loop_wrapper, this);
        if (event_thread_ == NULL)
          fprintf(stderr, "Unable to create thread: %s\n", SDL_GetError());
      }
    }
  }

  /** Return window width. */
  int window_width() const {
    return window_width_;
  }

  /** Return window height. */
  int window_height() const {
    return window_height_;
  }

  /** Erase graphics. */
  void clear() {
    safe_lock mutex(this);
    vis_.clear();
    label_ = std::string();
    request_render();
  }

  /** Runs the process specified in the constructor
   *  @param[in] process functor
   *  @param[in] start The virtual time the simulation starts at
   *  @param[in] end   The virtual time the simulation should end at
   *  @param[in] increment The change in time between each rerendering of the process
   *  @post process p's init is run once and the operator is run (end-start)/increment times
   */
  template <typename Process>
  void run(Process func, double start, double end, double increment) {

    // initialize the viewer for the set process
    CS207::Clock clock;
    func.init(this);

    // run the process iteratively in simulation
    double stop = 10;
    for (double t = start; t < end;) {
      if (!pause_mode_) {
        func(t, increment, this);
        t += increment;
      } else {
        CS207::sleep(.1);
      }
      if (quiet_ && clock.elapsed() > stop) {
        std::cout << "Virtual time elapsed: " << t << std::endl;
        break;
      }
    }
  }

  /** Adds an Action to be called specified key is pressed
   *  @param[in] act A pointer to the action to be added to the list of possible actions
   *  @post act is added to the list of possible actions during simulation
   */
  void add_action(Action* act, int id) {
    safe_lock mutex(this);
    actions_[id].push_back(act);
  }

  /** Removes an Action to be called specified key is pressed
   *  @param[in] act A pointer to the action to be removed from the list of possible actions
   *  @post act is removed froms the list of possible actions during simulation
   */
  void remove_action(Action* act, int id) {
    safe_lock mutex(this);
    actions_[id].erase(std::find(actions_[id].begin(), actions_[id].end(), act));
  }

  /** Add the nodes in the range [first, last) to the display.
   * @param[in] id Used to specify which object these nodes are for
   * @param[in] color_function Returns a Color for each node.
   * @param[in] position_function Returns a Point for each node.
   * @param[in] vector_function Returns a Point for each node.
   *
   * If @a id is -1, this is a new object, and id will be assigned to
   * a non-negative value.  Else, use the object with that id.
   * It's OK to add a Node more than once. Second and subsequent adds update
   * the existing node's position. */
  template <typename IT, typename C = DefaultColor, typename V = FalseFunctor, typename N = DefaultNormal>
    void add_nodes(IT first, IT last, int & id,
    N normal_function = N(), C color_function = C(), V vector_function = V()){
    // Lock for data update
    safe_lock mutex(this);
    assert(id < (int) vis_.size());
    if (id < 0) {
      id = vis_.size();
      vis_.push_back(Visualization());
      actions_.push_back(action_vec());
    }

    vis_[id].add_nodes(first, last, normal_function, color_function, vector_function);

    request_render();
  }

  /** Add the edges in the range [first, last) to the display.
   *
   * The InputIterator forward iterator must return edge objects. Given an
   * edge object e, the calls e.node1() and e.node2() must return its
   * endpoints.
   *
   * Edges whose endpoints weren't previously added by add_nodes() are
   * ignored. */
  template <typename IT>
  void add_edges(IT first, IT last, int id) {
    // Lock for data update
    safe_lock mutex(this);
    assert(id < (int) vis_.size() && id >= 0);

    vis_[id].add_edges(first, last);
    request_render();
  }

  /** Add the triangles in the range [first, last) to the display.
   *
   * The InputIterator forward iterator must return triangle objects. Given a
   * triangle object t, the calls t.node(i) for 0 <= i < 3 must return its
   * vertices.
   *
   * Triangles whose vertices weren't previously added by add_nodes() are
   * ignored. */
  template <typename IT>
  void add_triangles(IT first, IT last, int id) {
    // Lock for data update
    safe_lock mutex(this);
    assert(id < (int) vis_.size() && id >= 0);

    vis_[id].add_triangles(first, last);

    request_render();
  }

  /** Set a string label to display "green LCD" style. */
  void set_label(const std::string &str) {
    safe_lock mutex(this);
    if (str != label_) {
      label_ = str;
      request_render();
    }
  }

  /** Set a label to display "green LCD" style. */
  void set_label(double d) {
    std::stringstream ss;
    // At most 10 digits past the decimal point, but at most 4 trailing zeros
    ss << std::fixed << std::setprecision(10) << d;
    std::string s = ss.str();
    size_t dot;
    while ((dot = s.find('.')) != std::string::npos
      && dot + 5 < s.length()
      && s[s.length() - 1] == '0')
      s.erase(s.length() - 1);
    set_label(s);
  }

  /** Center view.
   *
   * Attempts to center the OpenGL view on the object by setting the new
   * viewpoint to the average of all nodes
   */
  void center_view() {
    {
      safe_lock mutex(this);
      Point center = vis_[cur_vis].center();
      // Set the new view point
      camera_.view_point(center.x, center.y, center.z);
    }

    // Queue for rendering
    request_render();
  }

  /** Request that the screen update shortly. */
  void request_render() {
    if (!render_requested_) {
      render_requested_ = true;
      // User event to force a render
      SDL_Event render_event;
      render_event.type = SDL_USEREVENT;
      render_event.user.code = 0;
      render_event.user.data1 = NULL;
      render_event.user.data2 = NULL;
      // Thread safe push event
      SDL_PushEvent(&render_event);
    }
  }

private:

  /** Initialize the SDL Window
   */
  void init() {
#if !__APPLE__
    // Set X11 Driver
    std::string driver = "SDL_VIDEODRIVER=x11";
    SDL_putenv((char*) driver.c_str());
#endif

    // Initialize SDL
    if (SDL_Init(SDL_INIT_EVERYTHING) < 0) {
      fprintf(stderr, "Init Error: %s\n", SDL_GetError());
      return;
    }

    window_width_ = 640;
    window_height_ = 480;

    int bits_per_pixel = 32;

    // Create the window and rendering surface
    surface_ = SDL_SetVideoMode(window_width_, window_height_, bits_per_pixel,
      SDL_HWSURFACE | SDL_GL_DOUBLEBUFFER | SDL_OPENGL);

    if (surface_ == NULL) {
      fprintf(stderr, "Display Error: %s\n", SDL_GetError());
      return;
    }

    // Background Color
    glClearColor(0.0, 0.0, 0.0, 1.0);

    // Initialize View
    glViewport(0, 0, window_width_, window_height_);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, window_width_ / (float) window_height_, 0.05, 1000.0);
    // Set up the camera view
    camera_.zoom_mag(2);

    // Point system
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // OpenGL Fog for better depth perception
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_FOG);
    glFogi(GL_FOG_MODE, GL_EXP);
    glFogf(GL_FOG_DENSITY, 0.3);


    GLTools::init_simple_light();
    GLTools::init_point_dist();

  }

  /** Static event loop wrapper for thread creation
   *
   * @param[in] _viewer_ptr Addtional data added when thread is launched.
   *                        Interpreted as an SDLViewer*
   * @return 0 Required by thread launcher
   */
  static int event_loop_wrapper(void* _viewer_ptr) {
    reinterpret_cast<Simulator*> (_viewer_ptr)->event_loop();
    return 0;
  }

  /** Main Render Loop
   *
   * Executed by the event thread until interrupt or killed
   */
  void event_loop() {
    {
      safe_lock mutex(this);
      init();
    }

    SDL_Event event;
    while (SDL_WaitEvent(&event) >= 0) {
      safe_lock mutex(this);
      handle_event(event);
    }
  }

  void select_nodes(bool select) {
    GLdouble model_view[16];
    GLdouble projection[16];
    GLint viewport[4];
    glGetDoublev(GL_MODELVIEW_MATRIX, model_view);
    glGetDoublev(GL_PROJECTION_MATRIX, projection);
    glGetIntegerv(GL_VIEWPORT, viewport);

    Point lower = Point(std::min(last_x_pos_, select_x_pos_), std::min(last_y_pos_, select_y_pos_), 0);
    Point upper = Point(std::max(last_x_pos_, select_x_pos_), std::max(last_y_pos_, select_y_pos_), 0);
    Point proj;

    //using vector projection to find point
    Visualization & v = vis_[cur_vis];
    typename std::vector<Point>::iterator it = v.coords_.begin(), end = v.coords_.end();
    for (; it != end; ++it) {
      Point & pos = *it;
      gluProject(pos.x, pos.y, pos.z, model_view, projection, viewport, &proj.x, &proj.y, &proj.z);
      if (proj.x < upper.x && proj.x > lower.x && proj.y < upper.y && proj.y > lower.y) {
        unsigned idx = it - v.coords_.begin();
        if (select) {
          v.select_node(idx);
        } else {
          v.deselect_node(idx);
        }
      }
    }
  }

  /** Main Event Handler
   *
   * @param[in] _event The SDL mouse, keyboard, or screen event to be handled
   */
  void handle_event(SDL_Event event) {
    switch (event.type) {
      // The mouse moved over the screen
    case SDL_MOUSEMOTION:
    {
      if (!select_mode_) {
        // Left mouse button is down
        if (event.motion.state == SDL_BUTTON(1)) {
          camera_.rotate_x(0.01 * event.motion.yrel);
          camera_.rotate_y(-0.01 * event.motion.xrel);
          request_render();
        }
        // Right mouse button is down
        if (event.motion.state == SDL_BUTTON(3)) {
          camera_.pan(-0.004 * event.motion.xrel, 0.004 * event.motion.yrel, 0);
          request_render();
        }
      }
      if (select_mode_ && (left_down_ || right_down_)) {
        request_render();
      }

      last_x_pos_ = (double) event.motion.x;

      last_y_pos_ = (double) (window_height_ - event.motion.y);

      // Avoid rendering on every mouse motion event
    }
      break;

    case SDL_MOUSEBUTTONDOWN:
    {
      // Wheel event
      if (event.button.button == SDL_BUTTON_WHEELUP)
        camera_.zoom(1.25);
      if (event.button.button == SDL_BUTTON_WHEELDOWN)
        camera_.zoom(0.8);
      if (select_mode_) {

        select_x_pos_ = (double) event.motion.x;
        select_y_pos_ = (double) (window_height_ - event.motion.y);
      }
      if (event.button.button == SDL_BUTTON_LEFT) {
        left_down_ = true;
      }
      if (event.button.button == SDL_BUTTON_RIGHT) {
        right_down_ = true;
      }

      request_render();
      break;
    }
    case SDL_MOUSEBUTTONUP:
    {
      if (select_mode_) {
        if (event.button.button == SDL_BUTTON_LEFT)
          select_nodes(true);
        if (event.button.button == SDL_BUTTON_RIGHT)
          select_nodes(false);
      }
      if (event.button.button == SDL_BUTTON_LEFT) {
        left_down_ = false;
      }
      if (event.button.button == SDL_BUTTON_RIGHT) {
        right_down_ = false;
      }

      request_render();
    }
      break;

    case SDL_KEYDOWN:
    {

      //check if any action uses one of these keys and if so, act on the node selected
      for (auto it = actions_.begin(); it != actions_.end(); ++it) {
        action_vec & actions = *it;
        for (unsigned i = 0; i < actions.size(); ++i) {
          if (event.key.keysym.sym == actions[i]->key) {
            actions[i]->act(vis_[cur_vis].selected_nodes_idxs.begin(), vis_[cur_vis].selected_nodes_idxs.end());
          }
        }
      }
      // Keyboard 'c' to center
      switch (event.key.keysym.sym) {
      case SDLK_c:
        center_view();
        break;
      case SDLK_p:
        pause_mode_ = !pause_mode_;
        break;
      case SDLK_v:
        vector_mode_ = !vector_mode_;
        break;
      case SDLK_s:
        select_mode_ = !select_mode_;
        break;
      case SDLK_t:
        triangle_mode_ = !triangle_mode_;
        break;
      case SDLK_d:
        draw_mode_ = !draw_mode_;
        break;
      case SDLK_LEFT:
        cur_vis = (cur_vis - 1 + vis_.size()) % vis_.size();
        break;
      case SDLK_RIGHT:
        cur_vis = (cur_vis - 1 + vis_.size()) % vis_.size();
        break;
      case SDLK_ESCAPE:
      case SDLK_q:
        // Keyboard 'esc' to exit
        exit(0);
      default:
        break;
      }
      request_render();
    }
      break;

    case SDL_KEYUP:
    {
    }
      break;

    case SDL_VIDEOEXPOSE:
    case SDL_USEREVENT:
    {
      // Render event
      render();
    }
      break;

      // Window 'x' to close
    case SDL_QUIT:
    {
      exit(0);
    }
      break;
    }
  }

  /** Render an "LCD digit" specified by the on bits in @a segments.
   *
   * Renders to the rectangle [0,0,1,2]. */
  void render_lcd_segments(unsigned segments) {
    glBegin(GL_LINES);
    for (int h = 0; h < 3; ++h, segments >>= 3) {
      if (segments & 1) {
        glVertex2f(0, h);
        glVertex2f(1, h);
      }
      if (segments & 2) {
        glVertex2f(0, h);
        glVertex2f(0, h + 1);
      }
      if (segments & 4) {
        glVertex2f(1, h);
        glVertex2f(1, h + 1);
      }
    }
    glEnd();
  }

  /** Renders a label in "green LCD" style. Only knows digits and '.'. */
  void render_label() {
    static const GLfloat skewscalem[] = {
      8, 0, 0, 0, 1.2, 8, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1
    };
    static const unsigned digit_segments[] = {
      0167, 0044, 0153, 0155, 0074, 0135, 0137, 0144, 0177, 0175
    };

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_FOG);

    // Set both relevant matrices for 2D display. The projection matrix is
    // orthographic; the model view matrix is the identity.
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, window_width_, 0, window_height_);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    //Draw selection box
    if (select_mode_ && (left_down_ || right_down_)) {
      if (right_down_) {
        glColor4f(0.5, 0, 0, 0.5);
      } else {
        glColor4f(0, 0.5, 0, 0.5);
      }
      glBegin(GL_QUADS);
      glVertex2f(select_x_pos_, select_y_pos_);
      glVertex2f(select_x_pos_, last_y_pos_);
      glVertex2f(last_x_pos_, last_y_pos_);
      glVertex2f(last_x_pos_, select_y_pos_);
      glEnd();
    }

    // Draw in green with fat points & lines.
    glColor3f(0, .759, .437);
    glPointSize(3);
    glLineWidth(1.5);

    // Estimate width.
    double expected_width = 0;
    for (std::string::const_iterator it = label_.begin(); it != label_.end(); ++it)
      expected_width += (*it == '.' ? 0.6 : (*it == ' ' ? 0.8 : 1.6));
    expected_width = roundf(std::max(expected_width * 8 + 10.0, 90.0));
    // Translate and skew.
    glTranslatef(window_width_ - expected_width + 0.375, 10.375, 0);
    glMultMatrixf(skewscalem);

    // Draw label.
    for (std::string::const_iterator it = label_.begin(); it != label_.end(); ++it)
      if (*it >= '0' && *it <= '9') {
        render_lcd_segments(digit_segments[*it - '0']);
        glTranslatef(1.6, 0, 0);
      } else if (*it == '.') {
        glBegin(GL_POINTS);
        glVertex2f(0, 0);
        glEnd();
        glTranslatef(0.6, 0, 0);
      } else if (*it == ' ')
        glTranslatef(0.8, 0, 0);
      else {
        render_lcd_segments(0001);
        glTranslatef(1.6, 0, 0);
      }


    glLoadIdentity();
    glTranslatef(10, window_height_ - 30, 0);
    glMultMatrixf(skewscalem);
    glColor3f(.9, .759, .437);
    render_lcd_segments(digit_segments[cur_vis]);


    // Restore settings.
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_FOG);
    GLTools::check_gl_error();
  }

  /** Render to the screen
   *
   * Double buffered rendering to the SDL Surface using OpenGL
   */
  void render() {
    safe_lock mutex(this);

    // Clear the screen and z-buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Construct the view matrix
    camera_.set_GLView();

    for (std::vector<Visualization>::iterator it = vis_.begin(); it != vis_.end(); ++it) {
      it->render(vector_mode_, draw_mode_, triangle_mode_);
    }
    // Draw the label
    if (!label_.empty())
      render_label();

    // Make visible
    SDL_GL_SwapBuffers();

    render_requested_ = false;
  }

  //To allow for pause mode
  bool pause_mode_;

  //To allow for vector display mode
  bool vector_mode_;

  //to allow for selecting, shading etc
  bool select_mode_;
  bool draw_mode_;
  bool triangle_mode_;

  //to designate the last positions the mouse passed over
  double last_x_pos_;
  double last_y_pos_;

  //to designate the location of the mousedown for selection
  double select_x_pos_;
  double select_y_pos_;
  double left_down_;
  double right_down_;

  //the visualization of a mesh/graph
  std::vector<Visualization> vis_;
  int cur_vis;

  //vector of actions added by the callback listener
  typedef std::vector<Action*> action_vec;
  std::vector<action_vec> actions_;

};


#endif
