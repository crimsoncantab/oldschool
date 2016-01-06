<?
    //require common code
    require_once("inc/common.inc");

    //inserts new athlete into coach's table in database    
    $sql = sprintf("INSERT INTO ". $_SESSION["username"] . " (athlete, pref, pretime, posttime) VALUES ('" . $_POST["athlete"] . "', '" . $_POST["pref"] . "', '0.00', '7.00')");

    mysql_query($sql);

    redirect("settings.php");

?>
