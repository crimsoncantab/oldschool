#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

//highest number of routers in network
#define MAX_ROUTER 20
//maximum # of reject/prefer policies
#define MAX_REJECT 5
#define MAX_PREFER 5
//host where everything runs
#define HOST "localhost"
//placeholder when id is not known
#define UNKNOWN_ID -1
//placeholder when port is not known
#define UNKNOWN_PORT 0
//how long is a timeout
#define TIMEOUT_SEC 1
//how many many timeouts between each link check/state update
#define NUM_TIMEOUT_WAIT 2
//size of buffer reading stdin
#define BUFFER_SIZE 256
//life of a message
#define TTL 20
//maximum length for border router peering key
#define MAX_KEY_LENGTH 8

//typedefs for convenience
typedef struct sockaddr_in addr;
typedef in_port_t port;
typedef char router_id;

typedef enum boolean_ {
    false = 0, true = 1
} boolean;

//keeps track of neighbors state

typedef struct neighbor_ {
    port n_port;
    router_id n_id;
    boolean alive;
} neighbor;

//a router's link state

typedef struct state_ {
    boolean linked[MAX_ROUTER];
} state;

//data to be send about a routers link state

typedef struct state_message_ {
    router_id s_id;
    state s;
} state_message;

//message data

typedef struct message_ {
    router_id dest_id;
    int ttl;
} message;

//data part of a packet

typedef union pack_data_ {
    state_message s;
    message m;
} pack_data;

//describes types of packets

typedef enum pack_type_ {
    stat,
    mess
} pack_type;

//packet metadata and payload

typedef struct packet_ {
    router_id from_id;
    port from_port;
    pack_type type;
    pack_data data;
} packet;

typedef struct border_peer_ {
    char peer_key[MAX_KEY_LENGTH];
    port peer_port;
    boolean alive;
} border_peer;

typedef union border_pack_data_ {
    message m;
    router_id pv[MAX_ROUTER][MAX_ROUTER];

} border_pack_data;

typedef struct border_packet_ {
    router_id from_id;
    char key[MAX_KEY_LENGTH];
    border_pack_data data;
    pack_type type;
} border_packet;

typedef struct prefer_ {
    int len;
    router_id path[MAX_ROUTER];
} prefer;

//routing table
router_id route[MAX_ROUTER][MAX_ROUTER];
//graph of topology
state as_topology[MAX_ROUTER];
//list of neighbors
neighbor * neighbors;
//num of neighbors
int valence;
//this router's unique id
router_id id;
//link state listening port for router
port ls_rec_port;
//listening socket
int ls_rec_socket;
//listening address
addr ls_rec_addr;
//address object for sending data
addr send_addr;
//boolean to check if shortest path needs to be rerun
boolean changed = false;

//for border routers
//is a border router
boolean is_border = false;
//path vector listening port for router
port pv_rec_port;
//path vector listening socket
int pv_rec_socket;
//path vector listening address
addr pv_rec_addr;
//this border router's peers
border_peer * peers[MAX_ROUTER];
//accessible destinations outside AS
//should be refreshed with every PV update
state extra_as_links;
//reject policies
int num_reject = 0;
router_id reject[MAX_REJECT];
//prefer policies
int num_prefer = 0;
prefer prefer_pol[MAX_PREFER];

//opens a socket, with error checking

int get_soc() {
    int s = socket(AF_INET, SOCK_DGRAM, 0);
    if (s < 0) {
        perror("socket");
        exit(1);
    }
    return s;
}

//send a border packet to address using socket, with error checking

void border_send_packet(border_packet * p, addr * address) {
    /*
        printf("sending packet %d, %d\n", sizeof(border_packet), address->sin_port);
     */
    int socket = get_soc();
    if (sendto(socket, p, sizeof (border_packet), 0,
            (struct sockaddr *) address, sizeof (addr)) < 0) {
        perror("sendto");
        exit(1);
    }
    close(socket);
}

//send a packet to address using socket, with error checking

void send_packet(packet * p, addr * address) {
    /*
        printf("sending packet\n");
     */
    int socket = get_soc();
    if (sendto(socket, p, sizeof (packet), 0,
            (struct sockaddr *) address, sizeof (addr)) < 0) {
        perror("sendto");
        exit(1);
    }
    close(socket);
}

//recieve border packet to listening address, with error checking

boolean border_recieve_packet(border_packet * p) {
    int len = sizeof (addr);
    int pack_size = recvfrom(pv_rec_socket, p, sizeof (border_packet), 0,
            (struct sockaddr *) & pv_rec_addr, &len);
    if (pack_size < 0) {
        perror("recvfrom");
        exit(1);
    }
    if (pack_size == sizeof (border_packet)) {
        return true;
    }
    return false;
}

//recieve packet to listening address, with error checking

boolean recieve_packet(packet * p) {
    int len = sizeof (addr);
    int pack_size = recvfrom(ls_rec_socket, p, sizeof (packet), 0,
            (struct sockaddr *) & ls_rec_addr, &len);
    if (pack_size < 0) {
        perror("recvfrom");
        exit(1);
    }
    if (pack_size == sizeof (packet)) {
        return true;
    }
    return false;
}

//lookup next hop for destination
//returns UNKNOWN_ID if router not found

router_id next_hop(router_id dest) {
    return route[dest][0];
}

//checks if route is intra-as

boolean border_in_as(router_id router) {
    int i;
    for (i = 0; i < MAX_ROUTER; i++) {
        if (as_topology[i].linked[router] == true)
            return true;
    }
    return false;
}

//resets routing table before running shortest path
//border routers only remove AS routers from table

void reset_route() {
    int i;
    for (i = 0; i < MAX_ROUTER; i++) {
        if (!is_border || border_in_as(i)) {
            int j;
            for (j = 0; j < MAX_ROUTER; j++) {
                route[i][j] = UNKNOWN_ID;
            }
        }
    }
}

//parse args. for simplicity, DOES NOT CHECK INVALID ARGS

void parse_args(int argc, char**argv) {
    int arg_i = 1;

    //do border router stuff
    if (strcmp(argv[arg_i], "-b") == 0) {
        printf("border router\n");
        is_border = true;
        /*
                char * val = argv[++arg_i];
         */
        pv_rec_port = htons(atoi(argv[++arg_i]));
        arg_i++;
    }

    id = atoi(argv[arg_i++]);
    /*
        printf("Id: %d\n", id);
     */

    ls_rec_port = htons(atoi(argv[arg_i++]));

    valence = argc - arg_i;
    /*
        printf("Valence: %d\n", valence);
     */

    neighbors = calloc(valence, sizeof (neighbor));
    int i;
    for (i = 0; i < valence; i++) {
        port n_port = atoi(argv[arg_i++]);
        /*
                printf("Neighbor\nPort: %d\n", n_port);
         */
        neighbors[i].n_port = (port) htons(n_port);
        neighbors[i].n_id = UNKNOWN_ID;
        neighbors[i].alive = false;
    }
}

//initialize router state

void init(int argc, char**argv) {
    parse_args(argc, argv);

    int i, j;
    for (i = 0; i < MAX_ROUTER; i++) {
        for (j = 0; j < MAX_ROUTER; j++) {
            as_topology[i].linked[j] = false;
            route[i][j] = UNKNOWN_ID;
        }
    }

    struct hostent *hp;
    hp = gethostbyname(HOST);
    if (hp == 0) {
        fprintf(stderr, "no host %s\n", argv[1]);
        exit(1);
    }

    ls_rec_socket = get_soc();

    ls_rec_addr.sin_family = AF_INET;
    memcpy(&ls_rec_addr.sin_addr, hp->h_addr, sizeof (ls_rec_addr.sin_addr));
    ls_rec_addr.sin_port = ls_rec_port;

    if (bind(ls_rec_socket, (struct sockaddr *) & ls_rec_addr,
            sizeof (ls_rec_addr)) < 0) {
        perror("bind");
        exit(1);
    }

    send_addr.sin_family = AF_INET;
    memcpy(&send_addr.sin_addr, hp->h_addr, sizeof (send_addr.sin_addr));

    if (is_border) {
        /*
                printf("initing border stuff\n");
         */
        memset(peers, 0, sizeof (peers));
        pv_rec_socket = get_soc();
        pv_rec_addr.sin_family = AF_INET;
        memcpy(&pv_rec_addr.sin_addr, hp->h_addr, sizeof (pv_rec_addr.sin_addr));
        pv_rec_addr.sin_port = pv_rec_port;

        if (bind(pv_rec_socket, (struct sockaddr *) & pv_rec_addr,
                sizeof (pv_rec_addr)) < 0) {
            perror("bind");
            exit(1);
        }
        /*
                printf("Bound to port %d\n", pv_rec_addr.sin_port);
         */
    }
}

//lookup a neighbor's port

void print_neighbors() {
    int i;
    for (i = 0; i < MAX_ROUTER; i++) {
        if (route[i][0] == i) {
            printf("%d ", i);
        }
    }
    printf("\n");
}

//print AS topology

void print_topo(state * top) {
    int i;
    for (i = 0; i < MAX_ROUTER; i++) {
        int j;
        for (j = 0; j < MAX_ROUTER; j++) {
            printf("%d ", top[i].linked[j]);
        }
        printf("\n");
    }
}

//print routing table

void print_route() {
    int i;
    for (i = 0; i < MAX_ROUTER; i++) {
        if (route[i][0] != UNKNOWN_ID) {
            int j;
            for (j = 0; route[i][j] != UNKNOWN_ID; j++) {
                printf("%d ", route[i][j]);
            }
            printf("\n");
        }
    }
}

//print all of the border's peers

void print_peers() {
    int i;
    for (i = 0; i < MAX_ROUTER; i++) {
        if (peers[i] != NULL) {
            printf("%d,%s\n", i, peers[i]->peer_key);
        }
    }

}

//lookup port of a neighbor by its id
//returns UNKNOWN_PORT if id is not a known neighbor

port port_from_id(router_id neighbor_id) {
    if (neighbor_id == UNKNOWN_ID) {
        return UNKNOWN_PORT;
    }
    int i;
    for (i = 0; i < valence; i++) {
        if (neighbors[i].n_id == neighbor_id) {
            return neighbors[i].n_port;
        }
    }
    return UNKNOWN_PORT;
}

//send a packet to a port

void prepare_packet(packet * p) {
    p->from_id = id;
    p->from_port = ls_rec_port;
}

//send a packet to a neighbor by id
//does nothing if neighbor's id is not mapped to a port

void unicast(packet * p, router_id neighbor_id) {
    prepare_packet(p);
    port n_port = port_from_id(neighbor_id);
    if (n_port != UNKNOWN_PORT) {
        send_addr.sin_port = n_port;
        send_packet(p, &send_addr);
    }
}

//sends a packet to a given router
//does nothing (drops packet) if no route known

void send_message(message * m) {

    router_id hop = next_hop(m->dest_id);
    if (is_border && peers[hop] != NULL) {
        border_packet bp;
        memcpy(&bp.data.m, m, sizeof (bp.data.m));
        send_addr.sin_port = peers[hop]->peer_port;
        memcpy(bp.key, peers[hop]->peer_key, sizeof (bp.key));
        bp.type = mess;
        bp.from_id = id;
        border_send_packet(&bp, &send_addr);
    } else if (hop != UNKNOWN_ID) {
        packet p;
        memcpy(&p.data.m, m, sizeof (message));
        p.type = mess;
        unicast(&p, hop);
    }
}

void process_message(message * m) {
    printf("%d\n", m->dest_id);
    if (--(m->ttl) && m->dest_id != id) {
        send_message(m);
    }
}

//put this router's id as the first hop in path (before sending on)

void append_self_to_path(router_id * path) {
    int i;
    for (i = MAX_ROUTER - 1; i > 0; i--) {
        path[i] = path[i - 1];
    }
    path[0] = id;

}

//broadcast path vector to peer border routers

void border_broadcast_pv() {
    /*
        printf("broadcasting pv\n");
     */
    border_packet p;
    memcpy(&(p.data.pv), route, sizeof (route));

    int i;
    //append this router to pv
    for (i = 0; i < MAX_ROUTER; i++) {
        if (p.data.pv[i][0] != UNKNOWN_ID || i == id) {
            append_self_to_path(p.data.pv[i]);
        }
    }

    p.from_id = id;
    p.type = stat;
    for (i = 0; i < MAX_ROUTER; i++) {
        if (peers[i] != NULL) {
            send_addr.sin_port = peers[i]->peer_port;
            memcpy(p.key, peers[i]->peer_key, sizeof (p.key));
            /*
                        printf("to someone: %d\n", send_addr.sin_port);
             */
            border_send_packet(&p, &send_addr);
        }
    }

}

//send a packet to *all* AP neighbors, regardless of status

void broadcast(packet * p) {
    prepare_packet(p);
    int i;
    for (i = 0; i < valence; i++) {
        send_addr.sin_port = neighbors[i].n_port;
        send_packet(p, &send_addr);
    }
}

//get the AS's perspective of what the border router links to
//(everthing outside AS)

void border_get_as_state(state * s) {
    int i;
    for (i = 0; i < MAX_ROUTER; i++) {
        if (as_topology[id].linked[i]) {
            //an AS neighbor
            s->linked[i] = true;
        } else if (border_in_as(i)) {
            //in as, but not a neighbor
            s->linked[i] = false;
        } else if (route[i][0] == UNKNOWN_ID) {
            //not connected to network
            s->linked[i] = false;
        } else {
            //on a different AS
            s->linked[i] = true;
        }

    }
}

//send a ping to neighbors

void broadcast_state() {
    packet p;
    p.type = stat;
    p.data.s.s_id = id;
    if (is_border) {
        //we want AS to think border router is neighbor of everything outside
        border_get_as_state(&(p.data.s.s));
    } else {
        p.data.s.s = as_topology[id];
    }
    broadcast(&p);
}

//generates a message

void gen_message(router_id dest) {
    message m;
    m.dest_id = dest;
    m.ttl = TTL;
    send_message(&m);
}

//remove router i from topology t

void delete_edges_to(router_id i, state * t) {
    int j;
    for (j = 0; j < MAX_ROUTER; j++) {
        t[j].linked[i] = false;
    }
}

#define ADD(v, n) n = (1 << v) | n
#define IN(v, n) (n & (1 << v))

//calculate the paths for the routing table

void run_shortest_path() {
    /*
        printf("Running SP\n");
     */
    reset_route();
    state temp_topology[MAX_ROUTER];
    memcpy(temp_topology, as_topology, sizeof (as_topology));

    //remove this router from topology
    delete_edges_to(id, temp_topology);

    int32_t n_prev = 0, n_new = 0;

    //begin by finding this router's neighbors
    ADD(id, n_prev);
    int depth = 0;
    //until there's no nodes left
    while (n_prev != 0) {
        int i, j;
        //iterate over all routers
        for (i = 0; i < MAX_ROUTER; i++) {
            //a router added in last iteration
            if (IN(i, n_prev)) {
                /*
                                printf("Checking neighbors of router %d\n", i);
                 */
                for (j = 0; j < MAX_ROUTER; j++) {
                    //found neighbor
                    if (temp_topology[i].linked[j]) {
                        /*
                                                printf("Router %d linked to router %d\n", i, j);
                         */
                        //copy current node's routing path to neighbor
                        memcpy(&(route[j][0]), &(route[i][0]), depth * sizeof (router_id));
                        //add neighbor to end of path
                        route[j][depth] = j;
                        //remove neighbor from topology
                        delete_edges_to(j, temp_topology);
                        //add neighbor to n_new
                        ADD(j, n_new);
                    }
                }
            }
        }
        depth++;
        //set n_prev to n_new
        n_prev = n_new;
        n_new = 0;
    }
    /*
        print_topo(temp_topology);
     */
}

//return the length of this path

int path_len(router_id * path) {
    int i;
    for (i = 0; path[i] != UNKNOWN_ID; i++);
    return i;
}

boolean path_contains_id(router_id * path, router_id val, int len) {
    int i;
    for (i = 0; i < len; i++) {
        if (path[i] == val) return true;
    }
    return false;
}

boolean border_rejected(router_id * path, int len) {
    int i;
    for (i = 0; i < num_reject; i++) {
        if (path_contains_id(path, reject[i], len)) return true;
    }
    return false;
}

void print_path(router_id * path, int len) {
    int i;
    for (i = 0; i < len; i++) {
        printf("%d ", path[i]);
    }
    printf("\n");
}

int border_prefer(router_id * path, int len) {
    int i;
    int highest = 0;
    for (i = 0; i < num_prefer; i++) {
        int j;
        boolean match = true;
        int min_len = (prefer_pol[i].len < len) ? prefer_pol[i].len : len;
        /*
                print_path(prefer_pol[i].path, prefer_pol[i].len);
         */
        for (j = 0; j < min_len; j++) {
            if (prefer_pol[i].path[j] != path[j]) {
                match = false;
                break;
            }
        }
        if (match) highest = i + 1;
    }
    return highest;
}

void border_accept(router_id * path, router_id dest, int len) {
    int i;
    for (i = 0; i < len; i++) {
        route[dest][i] = path[i];
    }
    for (; i < MAX_ROUTER; i++) {
        route[dest][i] = UNKNOWN_ID;
    }

}

//updates this border router's path vector

boolean border_update_pv(router_id * new_path, router_id peer_id, router_id dest) {
    int new_len = path_len(new_path);
    /*
        print_path(new_path, new_len);
     */
    if (dest == id) return false;
    router_id * old_path = route[dest];
    int old_len = path_len(old_path);
    if (border_rejected(new_path, new_len)) {
        /*
                printf("rejected\n");
         */
        return false;
    }
    if (path_contains_id(new_path, id, new_len)) {
        /*
                printf("self\n");
         */
        return false;
    }
    if (old_path[0] == peer_id) {
        /*
                printf("peer\n");
         */
        border_accept(new_path, dest, new_len);
        return true;
    }
    if (old_len == 0) {
        /*
                printf("infty\n");
         */
        border_accept(new_path, dest, new_len);
        return true;
    }
    int prefer_new = border_prefer(new_path, new_len);
    int prefer_old = border_prefer(old_path, old_len);

    /*
        printf("new %d, old %d\n", prefer_new, prefer_old);
     */
    if (prefer_old && !prefer_new) {
        /*
                printf("not preferred\n");
         */
        return false;
    }
    if (!prefer_old && prefer_new) {
        /*
                printf("preferred\n");
         */
        border_accept(new_path, dest, new_len);
        return true;
    }
    if (prefer_old && prefer_new) {
        if (prefer_old < prefer_new) {
            border_accept(new_path, dest, new_len);
            /*
                        printf("preferred\n");
             */
            return true;
        } else {
            /*
                        printf("not preferred\n");
             */
            return false;
        }
    }
    if (new_len < old_len && new_len != 0) { //0 means infty
        border_accept(new_path, dest, new_len);
        /*
                printf("shorter\n");
         */
        return true;
    }
    /*
        printf("longer\n");
     */

    return false;
}

//updates this router's topology with the state of another router

boolean update_topology(state * s, int router_id) {
    /*
        printf("updating topology\n");
     */
    if (router_id == id) return false;
    boolean top_changed = false;
    int i;
    state * old = &(as_topology[router_id]);
    for (i = 0; i < MAX_ROUTER; i++) {
        //state has changed
        if (old->linked[i] != s->linked[i]) {
            //update state
            old->linked[i] = s->linked[i];
            //propogate change
            top_changed = true;
            /*
                        printf("found change\n");
             */
        }
    }
    return top_changed;
}

//check if peers are responsive

boolean border_check_peers() {
    int i;
    boolean neighbor_changed = false;
    for (i = 0; i < MAX_ROUTER; i++) {
        if (peers[i] != NULL) {
            if (!peers[i]->alive) {
                neighbor_changed = true;
                int j;
                for (j = 0; j < MAX_ROUTER; j++) {
                    if (route[j][0] == i) {
                        memset(route[j], UNKNOWN_ID, MAX_ROUTER * sizeof (router_id));
                    }
                }
            }
            peers[i]->alive = false;
        }
    }
    return neighbor_changed;
}

//updates this router's state based on responsiveness of neighbors

boolean update_state() {
    int i;
    boolean neighbor_changed = false;
    for (i = 0; i < valence; i++) {
        //router hasn't shown up yet
        if (neighbors[i].n_id == UNKNOWN_ID) {
            continue;
        }
        state * s = &as_topology[id];
        //state has changed
        if (neighbors[i].alive != s->linked[neighbors[i].n_id]) {
            //update state
            s->linked[neighbors[i].n_id] = neighbors[i].alive;
            neighbor_changed = true;
        }
        //neighbor must ping by next update to stay alive
        neighbors[i].alive = false;
    }
    if (is_border) {
        neighbor_changed = neighbor_changed || border_check_peers();
    }
    return neighbor_changed;
}

//match keys

boolean keys_equal(char * key1, char * key2) {
    if (strncmp(key1, key2, MAX_KEY_LENGTH)) {
        return false;
    }
    return true;
}

//uses packet from border peer to mark it as alive

boolean border_update_neighbor(border_packet * p) {

    router_id peer = p->from_id;
    /*
        printf("recieved border packet from %d\n", peer);
     */
    //authenticate
    if (peers[peer] != NULL && keys_equal(p->key, peers[peer]->peer_key)) {
        peers[peer]->alive = true;
        return true;
    }
    return false;
}

//uses packet from neighbor to mark it as alive

boolean update_neighbor(packet * p) {
    /*
        printf("updating neighbor\n");
     */
    int i;
    for (i = 0; i < valence; i++) {
        if (neighbors[i].n_port == p->from_port) {
            neighbors[i].alive = true;
            neighbors[i].n_id = p->from_id;
            return true;
        }
    }
    return false;

}

//while parsing a string, go to index after space

char * skip_next_space(char * str) {
    int i = 0;
    while (str[i] != ' ') i++;
    return &(str[++i]);
}

//copy a key from src to dest

void set_key(char * dest, char * src) {
    int i;
    for (i = 0; src[i] != '\n' && src[i] != ' ' && i < MAX_KEY_LENGTH; i++) {
        dest[i] = src[i];
    }
    for (; i < MAX_KEY_LENGTH; i++) {
        dest[i] = '\0';
    }
}

//add a router to peer list

void border_add_peer(char * cmd) {
    router_id peer_id = atoi(cmd);
    peers[peer_id] = malloc(sizeof (border_peer));
    if (!peers[peer_id]) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    cmd = skip_next_space(cmd);
    peers[peer_id]->peer_port = htons(atoi(cmd));
    cmd = skip_next_space(cmd);
    set_key(peers[peer_id]->peer_key, cmd);
    peers[peer_id]->alive = false;
/*
    printf("Added peer %d, %s, %d\n", peer_id, peers[peer_id]->peer_key, peers[peer_id]->peer_port);
*/
}

//add a prefer policy

void border_add_prefer(char * cmd) {
    int n = atoi(cmd);
    n = (n > MAX_ROUTER) ? MAX_ROUTER : n;
    prefer_pol[num_prefer].len = n;
    int i;
    for (i = 0; i < n; i++) {
        cmd = skip_next_space(cmd);
        prefer_pol[num_prefer].path[i] = atoi(cmd);
    }
    num_prefer++;
}

//called when router times out

void handle_timeout(int num_timeouts) {

    if (num_timeouts == 0) {
        if (update_state() || changed) {
            run_shortest_path();
            changed = false;
        }
    }
    broadcast_state();
    if (is_border) border_broadcast_pv();
}

//called when recieving data from stdin

void handle_stdin() {
    char buf[BUFFER_SIZE];
    if (fgets(buf, sizeof (buf), stdin) == 0) {
        //eof
        exit(0);
    }
    router_id dest;

    if (is_border) {
        switch (buf[0]) {
            case 'S':
            case 's':
                border_add_peer(&(buf[2]));
                break;
            case 'P':
            case 'p':
                border_add_prefer(&(buf[2]));
                break;
            case 'R':
            case 'r':
                if (num_reject < MAX_REJECT) {
                    router_id rej_id = atoi(&(buf[2]));
                    reject[num_reject++] = rej_id;
                    int i;
                    for (i = 0; i < MAX_ROUTER; i++) {
                        if (border_rejected(route[i], path_len(route[i]))) {
                            memset(route[i], UNKNOWN_ID, MAX_ROUTER * sizeof (router_id));
                        }
                    }

                }
                break;
            case 'e':
            case 'E':
                print_peers();
                break;
        }
    }

    switch (buf[0]) {
        case 'N':
        case 'n':
            print_neighbors();
            break;
        case 'T':
        case 't':
            print_route();
            break;
            /*
                    case 'P':
                    case 'p':
                        print_topo(as_topology);
                        break;
             */
        case '0': //need this because atoi returns 0 on error
            gen_message(0);
        default:
            dest = atoi(buf);
            if (dest) {
                gen_message(dest);
            }
            break;
    }

}

//called when recieving packet on listing port

void handle_packet() {
    packet p;
    if (recieve_packet(&p) && update_neighbor(&p)) {

        /*
                printf("handling packet\n");
         */
        if (p.type == stat) {
            state_message s = p.data.s;
            if (update_topology(&(s.s), s.s_id)) {
                changed = true;
                broadcast(&p);
            }
        }
        else if (p.type == mess) {
            message m = p.data.m;

            process_message(&m);
        }
    }
}

//called when recieving packet from peer border router

void border_handle_packet() {
    border_packet p;
    if (border_recieve_packet(&p) && border_update_neighbor(&p)) {
        if (p.type == stat) {
            /*
                        printf("before update\n");
                        print_route();
             */
            boolean changed = false;
            int i;
            for (i = 0; i < MAX_ROUTER; i++) {
                /*
                                printf("%d ", i);
                 */
                if (border_update_pv(p.data.pv[i], p.from_id, i)) {
                    changed = true;
                }
            }
            /*
                        printf("after update\n");
                        print_route();
             */

            if (changed) {
                //our state has changed
                //do we need to broadcast?
                /*
                                border_broadcast_pv();
                 */
            }
        } else if (p.type == mess) {
            message m = p.data.m;
            process_message(&m);
        }
    }

}

//waits for activity on stdin or port

int wait(fd_set * mask, struct timeval * timer) {
    int range = ((ls_rec_socket < pv_rec_socket) ? pv_rec_socket : ls_rec_socket) + 1;
    int n = select(range, mask, 0, 0, timer);
    if (n < 0) {
        perror("select");
        exit(1);
    }
    return n;
}

//router looping function

void loop() {
    fd_set mask;
    struct timeval timer;
    timer.tv_sec = TIMEOUT_SEC;
    timer.tv_usec = 0;
    int num_timeouts = 0;
    while (1) {
        FD_ZERO(&mask);
        FD_SET(fileno(stdin), &mask);
        FD_SET(ls_rec_socket, &mask);
        if (is_border) {
            FD_SET(pv_rec_socket, &mask);
        }
        if (wait(&mask, &timer) == 0) {
            num_timeouts++;
            num_timeouts %= NUM_TIMEOUT_WAIT;
            handle_timeout(num_timeouts);
            timer.tv_sec = TIMEOUT_SEC;
            timer.tv_usec = 0;
        }
        if (FD_ISSET(fileno(stdin), &mask)) {
            handle_stdin();
        }
        if (FD_ISSET(ls_rec_socket, &mask)) {
            handle_packet();
        }
        if (is_border && FD_ISSET(pv_rec_socket, &mask)) {
            border_handle_packet();
        }
    }
}

int main(int argc, char**argv) {
    init(argc, argv);
    loop();
    return EXIT_SUCCESS;
}