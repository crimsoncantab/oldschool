<? 

    // require common code
    require_once("inc/common.inc"); 

    // destroy session, effectively logging user out
    session_destroy();

    redirect("index.php");

?>
