<?

    /***********************************************************************
     * helpers.inc
     * 
     * Adapted from:
     * Computer Science 50
     *
     * Helper functions.
     **********************************************************************/

    /*
     * void
     * apologize($message)
     *
     * Apologizes to user by displaying a page with message.
     */

    function apologize($message)
    {
        // require template
        require_once("apology.inc");

        // exit immediately since we're apologizing
        exit;
    }


    /*
     * void
     * dump($variable)
     *
     * Facilitates debugging by dumping contents of variable
     * to browser.
     */

    function dump($variable)
    {
        // dump variable using some quick and dirty (albeit invalid) XHTML
        if (!$variable)
            print("Variable is null or otherwise empty.");
        else
            print("<pre>" . print_r($variable, true) . "</pre>");

        // exit immediately so that we can see what we printed
        exit;
    }

    /*
     * void
     * redirect($destination)
     * 
     * Redirects user to destination, which can be
     * a URL or a relative path on the local host.
     *
     * Because this function outputs an HTTP header, it
     * must be called before caller outputs any XHTML.
     */

    function redirect($destination)
    {
        // handle URL
        if (preg_match("/^http:\/\//", $destination))
            header("Location: " . $destination);

        // handle absolute path
        else if (preg_match("/^\//", $destination))
        {
            $host = $_SERVER["HTTP_HOST"];
            header("Location: http://$host$destination");
        }

        // handle relative path
        else
        {
            // adapted from http://www.php.net/header
            $host = $_SERVER["HTTP_HOST"];
            $path = rtrim(dirname($_SERVER["PHP_SELF"]), "/\\");
            header("Location: http://$host$path/$destination");
        }

        // exit immediately since we're redirecting anyway
        exit;
    }

?>
