response.title = settings.title
response.subtitle = settings.subtitle
response.meta.author = '%s <%s>' % (settings.author, settings.author_email)
response.meta.keywords = settings.keywords
response.meta.description = settings.description
response.menu = [
    (T('Home'),False,URL('default','index'),[]),
    (T('Introduction'),False,URL('default','intro'),[]),
    (T('Exhibit'),False,URL('exhibit','index'),[
        (T('Catalog'),False,URL('exhibit','catalog'))
    ]),
    (T('Bibliography'),False,URL('default','bib'),[]),
#    (T('Home'),URL('default','index').xml()==URL().xml(),URL('default','index'),[]),
]
