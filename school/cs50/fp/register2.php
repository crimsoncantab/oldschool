<? 

    // require common code
    require_once("inc/common.inc"); 

    // escape username and password for safety
    $username = mysql_real_escape_string($_POST["username"]);
    $password = mysql_real_escape_string($_POST["password"]);
    $password2 = mysql_real_escape_string($_POST["password2"]);
    $name = mysql_real_escape_string($_POST["name"]);

    // check validity of username and password
    if ($username == "" || $password == "" || $name == "")
        apologize("Username, Name, and/or Password is blank.");
    if ($password != $password2)
        apologize("Passwords do not match");

    // prepare SQL
    $sql = sprintf("INSERT INTO users(username, password, name)
                   VALUE('%s', '%s', '%s')",
                   $username, $password, $name);

    // execute query
    $result = mysql_query($sql);

    // if insert successful, start session and create table
    if ($result)
    {

        mysql_query("CREATE TABLE " . $username . " (athlete varchar( 60 ) NOT NULL , pref enum( 'c', 'p', 's' ) NOT NULL , pretime decimal( 10, 2 ) NOT NULL default '0.00', posttime decimal( 10, 2 ) NOT NULL default '7.00', PRIMARY KEY ( `athlete` ) ) ENGINE = InnoDB DEFAULT CHARSET = latin1");

        // cache username in session
        $_SESSION["username"] = $username;

        // redirect to coach's site
        redirect("settings.php");
    }

    // else report error
    else
        apologize("Invalid username.");

?>
