<?
    // require common code
    require_once("inc/common.inc");
    
    //prepares information for e-mail to be sent
    $headers = 'From: mcginn@fas.harvard.edu' . "\r\n" .
        'Reply-To: mcginn@fas.harvard.edu' . "\r\n" .
            'X-Mailer: PHP/' . phpversion();

    //sends message, reports error on failure
    if(!mail($_POST["to"], $_POST["subject"], $_POST["message"], $headers))
        apologize("E-mail could not be sent");
    else
        //redirects on success
        redirect("settings.php");
?>
