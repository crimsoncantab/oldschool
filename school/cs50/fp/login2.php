<? 
    // require common code
    require_once("inc/common.inc"); 
    // escape username and password for safety
    $username = mysql_real_escape_string($_POST["username"]);
    $password = mysql_real_escape_string($_POST["password"]);
    
    // prepare SQL
    $sql = sprintf("SELECT username FROM users WHERE username='%s' AND password='%s'", $username, $password);

    // execute query
    $result = mysql_query($sql);

    // if we found a row, remember user and redirect to settings
    if (mysql_num_rows($result) == 1)
    {
        // grab row
        $row = mysql_fetch_array($result);

        // cache uid in session
        $_SESSION["username"] = $row["username"];

        // redirect to settings
        redirect("settings.php");
    }

    // else report error
    else
        apologize("Invalid username and/or password.");

?>
