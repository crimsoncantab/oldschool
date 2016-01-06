<?
    // require common code
    require_once("inc/common.inc");

    // retrieves posted variables
    $athlete = mysql_real_escape_string($_POST["athlete"]);
    $pretimeh = mysql_real_escape_string($_POST["pretimeh"]);
    $posttimeh = mysql_real_escape_string($_POST["posttimeh"]);
    $pretimem = mysql_real_escape_string($_POST["pretimem"]);
    $posttimem = mysql_real_escape_string($_POST["posttimem"]);

    //turns times into decimal values
    if ($pretimeh == 12) $pretimeh = 0;
    if ($posttimeh == 12) $posttimeh = 0;
    if ($posttimeh == 7) $posttimem = 00;
    
    $pretime = ($pretimeh + $pretimem / 60);
    $posttime = ($posttimeh + $posttimem / 60);
    
    if ($pretime > $posttime) {
        $pretime = 0;
        $posttime = 7;
    }
    
    //updates databse with new values and redirects on success
    $sql = sprintf("UPDATE " . $_SESSION["name"] . " SET pretime=" . $pretime . " WHERE athlete='" . $athlete . "'");
    $sql2 = sprintf("UPDATE " . $_SESSION["name"] . " SET posttime=" . $posttime . " WHERE athlete='" . $athlete . "'");
    
    if (mysql_query($sql) && mysql_query($sql2))
        redirect("index.php");
    else
        apologize("Could not update practice times.");
?>
