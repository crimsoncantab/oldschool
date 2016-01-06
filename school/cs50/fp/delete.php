<?
    // require common code
    require_once("inc/common.inc");
    
    // remove athlete from coach's table in database
    $sql = sprintf("DELETE FROM " . $_SESSION["username"] . " WHERE athlete = '" . $_POST["athlete"] . "' LIMIT 1");

    mysql_query($sql);

    redirect("settings.php");
?>
